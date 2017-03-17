#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trains and tests a DenseNet on CIFAR-10.

For usage information, call with --help.

Author: Jan Schlüter
"""

import os
from argparse import ArgumentParser


def opts_parser():
    usage = "Trains and tests a DenseNet on CIFAR-10."
    parser = ArgumentParser(description=usage)
    parser.add_argument(
            '-L', '--depth', type=int, default=40,
            help='Network depth in layers (default: %(default)s)')
    parser.add_argument(
            '-k', '--growth-rate', type=int, default=12,
            help='Growth rate in dense blocks (default: %(default)s)')
    parser.add_argument(
            '--dropout', type=float, default=0,
            help='Dropout rate (default: %(default)s)')
    parser.add_argument(
            '--augment', action='store_true', default=True,
            help='Perform data augmentation (enabled by default)')
    parser.add_argument(
            '--no-augment', action='store_false', dest='augment',
            help='Disable data augmentation')
    parser.add_argument(
            '--validate', action='store_true', default=False,
            help='Perform validation on validation set (disabled by default)')
    parser.add_argument(
            '--no-validate', action='store_false', dest='validate',
            help='Disable validation')
    parser.add_argument(
            '--validate-test', action='store_const', dest='validate',
            const='test', help='Perform validation on test set')
    parser.add_argument(
            '--epochs', type=int, default=300,
            help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
            '--eta', type=float, default=0.1,
            help='Initial learning rate (default: %(default)s)')
    parser.add_argument(
            '--save-weights', type=str, default=None, metavar='FILE',
            help='If given, save network weights to given .npz file')
    parser.add_argument(
            '--save-errors', type=str, default=None, metavar='FILE',
            help='If given, save train/validation errors to given .npz file')
    return parser


def generate_in_background(generator, num_cached=10):
    """
    Runs a generator in a background thread, caching up to `num_cached` items.
    """
    import Queue
    queue = Queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        item = queue.get()


def train_test(depth, growth_rate, dropout, augment, validate, epochs,
               eta, save_weights, save_errors, batchsize=64):
    # import (deferred until now to make --help faster)
    import numpy as np
    import theano
    import theano.tensor as T
    import lasagne

    import densenet_fast as densenet  # or "import densenet" for slower version
    import cifar10
    import progress

    # instantiate network
    print("Instantiating network...")
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = densenet.build_densenet(input_var=input_var, depth=depth,
                                      growth_rate=growth_rate, dropout=dropout)
    print("%d layers with weights, %d parameters" %
          (sum(hasattr(l, 'W')
               for l in lasagne.layers.get_all_layers(network)),
           lasagne.layers.count_params(network, trainable=True)))

    # load dataset
    print("Loading dataset...")
    X_train, y_train, X_test, y_test = cifar10.load_dataset(
            path=os.path.join(os.path.dirname(__file__), 'data'))
    if validate == 'test':
        X_val, y_val = X_test, y_test
    elif validate:
        X_val, y_val = X_train[-5000:], y_train[-5000:]
        X_train, y_train = X_train[:-5000], y_train[:-5000]

    # define training function
    print("Compiling training function...")
    prediction = lasagne.layers.get_output(network)
    # note: The Keras implementation clips predictions for the categorical
    #       cross-entropy. This doesn't seem to have a positive effect here.
    # prediction = T.clip(prediction, 1e-7, 1 - 1e-7)
    loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                       target_var).mean()
    # note: The paper says 1e-4 decay, but 1e-4 in Torch is 5e-5 elsewhere.
    #       However, 1e-4 seems to work better than 5e-5, so we use 1e-4.
    # note: Torch includes biases in L2 decay. This seems to be important! So
    #       we decay all 'trainable' parameters, not just 'regularizable' ones.
    l2_loss = 1e-4 * lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l2, {'trainable': True})
    params = lasagne.layers.get_all_params(network, trainable=True)
    eta = theano.shared(lasagne.utils.floatX(eta), name='eta')
    updates = lasagne.updates.nesterov_momentum(
            loss + l2_loss, params, learning_rate=eta, momentum=0.9)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    l2_fn = theano.function([], l2_loss)

    # define validation/testing function
    print("Compiling testing function...")
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var).mean()
    test_err = 1 - lasagne.objectives.categorical_accuracy(test_prediction,
                                                           target_var).mean(
                                                  dtype=theano.config.floatX)
    test_fn = theano.function([input_var, target_var], [test_loss, test_err])

    # Finally, launch the training loop.
    print("Starting training...")
    if save_errors:
        errors = []
    for epoch in range(epochs):
        # shrink learning rate at 50% and 75% into training
        if epoch == (epochs // 2) or epoch == (epochs * 3 // 4):
            eta.set_value(eta.get_value() * lasagne.utils.floatX(0.1))

        # In each epoch, we do a full pass over the training data:
        train_loss = 0
        train_batches = len(X_train) // batchsize
        batches = cifar10.iterate_minibatches(X_train, y_train, batchsize,
                                              shuffle=True)
        if augment:
            batches = cifar10.augment_minibatches(batches)
            batches = generate_in_background(batches)
        batches = progress.progress(
                batches, desc='Epoch %d/%d, Batch ' % (epoch + 1, epochs),
                total=train_batches)
        for inputs, targets in batches:
            train_loss += train_fn(inputs, targets)

        # And possibly a full pass over the validation data:
        if validate:
            val_loss = 0
            val_err = 0
            val_batches = len(X_val) // batchsize
            for inputs, targets in cifar10.iterate_minibatches(X_val, y_val,
                                                               batchsize,
                                                               shuffle=False):
                loss, err = test_fn(inputs, targets)
                val_loss += loss
                val_err += err

        # Then we print the results for this epoch:
        train_loss /= train_batches
        l2_loss = l2_fn()
        print("  training loss:\t%.6f" % train_loss)
        print("  L2 loss:      \t%.6f" % l2_loss)
        if save_errors:
            errors.extend([train_loss, l2_loss])
        if validate:
            val_loss /= val_batches
            val_err /= val_batches
            print("  validation loss:\t%.6f" % val_loss)
            print("  validation error:\t%.2f%%" % (val_err * 100))
            if save_errors:
                errors.extend([val_loss, val_err])

    # After training, we compute and print the test error:
    test_loss = 0
    test_err = 0
    test_batches = len(X_test) // batchsize
    for inputs, targets in cifar10.iterate_minibatches(X_test, y_test,
                                                       batchsize,
                                                       shuffle=False):
        loss, err = test_fn(inputs, targets)
        test_loss += loss
        test_err += err
    print("Final results:")
    print("  test loss:\t\t%.6f" % (test_loss / test_batches))
    print("  test error:\t\t%.2f%%" % (test_err / test_batches * 100))

    # Optionally, we dump the network weights to a file
    if save_weights:
        np.savez(save_weights, *lasagne.layers.get_all_param_values(network))

    # Optionally, we dump the learning curves to a file
    if save_errors:
        errors = np.asarray(errors).reshape(epochs, -1)
        np.savez(save_errors, errors=errors)


def main():
    # parse command line
    parser = opts_parser()
    args = parser.parse_args()

    # run
    train_test(**vars(args))


if __name__ == "__main__":
    main()

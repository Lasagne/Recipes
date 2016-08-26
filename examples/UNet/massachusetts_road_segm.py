__author__ = 'Fabian Isensee'
import numpy as np
import lasagne
import os
import sys
import fnmatch
import matplotlib.pyplot as plt
sys.path.append("../../modelzoo/")
from Unet import *
import theano.tensor as T
import theano
import cPickle
from time import sleep
from generators import batch_generator, threaded_generator, random_crop_generator
from massachusetts_road_dataset_utils import prepare_dataset

def plot_some_results(pred_fn, test_generator, BATCH_SIZE, PATCH_SIZE = 192, n_images=10):
    fig_ctr = 0
    for data, seg in test_generator:
        res = pred_fn(data)
        for d, s, r, p in zip(data, seg, res):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(d.transpose(1,2,0))
            plt.subplot(1, 3, 2)
            plt.imshow(s[0])
            plt.subplot(1, 3, 3)
            plt.imshow(r[0])
            plt.savefig("road_segmentation_result_%03.0f.png"%fig_ctr)
            plt.close()
            fig_ctr += 1
            if fig_ctr > n_images:
                break

def main():
    # only download dataset once. This takes a while.
    if not os.path.isfile("test_target.npy"):
        # heuristic that I included to make sure the dataset is only donwloaded and prepared once
        prepare_dataset()

    # set some hyper parameters. You should not have to touch anything if you have 4GB or more VRAM
    BATCH_SIZE = 24
    N_EPOCHS = 30
    N_BATCHES_PER_EPOCH = 100
    N_BATCHES_PER_EPOCH_valid = 15
    PATCH_SIZE = 128+64

    # load the prepared data. They have been converted to np arrays because they are much faster to load than single image files.
    # you will need some ram in order to have everything in memory.
    # If you are having RAM issues, change mmap_mode to 'r'. This will not load the entire array into memory but rather
    # read from disk the bits that we currently need
    # (if you have, copy your repository including the data to an SSD, otherwise it will take a long time to
    # generate batches)
    mmap_mode = None
    data_train = np.load("train_data.npy", mmap_mode=mmap_mode)
    target_train = np.load("train_target.npy", mmap_mode=mmap_mode)
    data_valid = np.load("valid_data.npy", mmap_mode=mmap_mode)
    target_valid= np.load("valid_target.npy", mmap_mode=mmap_mode)
    data_test = np.load("test_data.npy", mmap_mode=mmap_mode)
    target_test = np.load("test_target.npy", mmap_mode=mmap_mode)

    # we are using pad='same' for simplicity (otherwise we would have to crop our ground truth).
    net = build_UNet(n_input_channels=3, BATCH_SIZE=BATCH_SIZE, num_output_classes=2, pad='same',
                     nonlinearity=lasagne.nonlinearities.elu, input_dim=(PATCH_SIZE, PATCH_SIZE),
                     base_n_filters=16, do_dropout=False)
    output_layer_for_loss = net["output_flattened"]

    # this is np.sum(target_train == 0) and np.sum(target_train == 1). No need to compute this every time
    class_frequencies = np.array([2374093357., 118906643.])
    # we are taking the log here because we want the net to focus more on the road pixels but not too much (otherwise
    # it would not be penalized enough for missclassifying terrain pixels which results in too many false positives)
    class_weights = np.log(class_frequencies[[1,0]])
    class_weights = class_weights / np.sum(class_weights) * 2.
    class_weights = class_weights.astype(np.float32)

    # if you wish to load pretrained weights you can uncomment this code and modify the file name
    # if you want, use my pretained weights (got around 96% accuracy and a loss of 0.11 using excessive data
    # augmentation: cropping, rotation, elastic deformation)
    # https://www.dropbox.com/s/t6juf6o2ix7dntk/UNet_roadSegmentation_Params.zip?dl=0
    '''with open("UNet_params_ep0.pkl", 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)'''

    x_sym = T.tensor4()
    seg_sym = T.ivector()
    w_sym = class_weights[seg_sym]

    # add some weight decay
    l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss, lasagne.regularization.l2) * 1e-4

    # the distinction between prediction_train and test is important only if we enable dropout
    prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False)
    # we could use a binary loss but I stuck with categorical crossentropy so that less code has to be changed if your
    # application has more than two classes
    loss = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
    loss *= w_sym
    loss = loss.mean()
    loss += l2_loss
    acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym), dtype=theano.config.floatX)

    prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True)
    loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, seg_sym)

    # we multiply our loss by a weight map. In this example the weight map only increases the loss for road pixels and
    # decreases the loss for other pixels. We do this to ensure that the network puts more focus on getting the roads
    # right
    loss_val *= w_sym
    loss_val = loss_val.mean()
    loss_val += l2_loss
    acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym), dtype=theano.config.floatX)

    # learning rate has to be a shared variablebecause we decrease it with every epoch
    params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
    learning_rate = theano.shared(np.float32(0.001))
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

    # create a convenience function to get the segmentation
    seg_output = lasagne.layers.get_output(net["output_segmentation"], x_sym)
    seg_output = seg_output.argmax(1)

    train_fn = theano.function([x_sym, seg_sym], [loss, acc_train], updates=updates)
    val_fn = theano.function([x_sym, seg_sym], [loss_val, acc])
    get_segmentation = theano.function([x_sym], seg_output)

    # some data augmentation. If you want better results you should invest more effort here. I left rotations and
    # deformations out for the sake of speed and simplicity
    train_generator = random_crop_generator(batch_generator(data_train, target_train, BATCH_SIZE), PATCH_SIZE)
    train_generator = threaded_generator(train_generator, num_cached=10)

    # there is no need for data augmentation on the validation. However we need patches of the same size which is why
    # we are using the random crop generator here again
    validation_generator = random_crop_generator(batch_generator(data_valid, target_valid, BATCH_SIZE), PATCH_SIZE)
    validation_generator = threaded_generator(validation_generator, num_cached=10)

    # do the actual training
    for epoch in range(N_EPOCHS):
        print epoch
        losses_train = []
        n_batches = 0
        accuracies_train = []
        for data, target in train_generator:
            # the output of the net has shape (BATCH_SIZE, N_CLASSES). We therefore need to flatten the segmentation so
            # that we can match it with the prediction via the crossentropy loss function
            target_flat = target.flatten()
            loss, acc = train_fn(data.astype(np.float32), target_flat)
            losses_train.append(loss)
            accuracies_train.append(acc)
            n_batches += 1
            if n_batches > N_BATCHES_PER_EPOCH:
                break
        print "epoch: ", epoch, "\ntrain accuracy: ", np.mean(accuracies_train), " train loss: ", np.mean(losses_train)

        losses_val = []
        accuracies_val = []
        n_batches = 0
        for data, target in validation_generator:
            target_flat = target.flatten()
            loss, acc = val_fn(data.astype(np.float32), target_flat)
            losses_val.append(loss)
            accuracies_val.append(acc)
            n_batches += 1
            if n_batches > N_BATCHES_PER_EPOCH_valid:
                break
        print "val accuracy: ", np.mean(accuracies_val), " val loss: ", np.mean(losses_val)
        learning_rate *= 0.2
        # save trained weights after each epoch
        with open("UNet_params_ep%03.0f.pkl"%epoch, 'w') as f:
            cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)

    # create some png files showing (raw image, ground truth, prediction). Of course we use the test set here ;-)
    test_gen = random_crop_generator(batch_generator(data_test, target_test, BATCH_SIZE), PATCH_SIZE)
    plot_some_results(get_segmentation, test_gen, BATCH_SIZE)


if __name__ == "__main__":
    main()
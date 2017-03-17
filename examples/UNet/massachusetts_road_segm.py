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
from generators import batch_generator, threaded_generator, random_crop_generator, center_crop_generator
from massachusetts_road_dataset_utils import prepare_dataset
from sklearn.metrics import roc_auc_score

def plot_some_results(pred_fn, test_generator, n_images=10):
    fig_ctr = 0
    for data, seg in test_generator:
        res = pred_fn(data)
        for d, s, r in zip(data, seg, res):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(d.transpose(1,2,0))
            plt.title("input patch")
            plt.subplot(1, 3, 2)
            plt.imshow(s[0])
            plt.title("ground truth")
            plt.subplot(1, 3, 3)
            plt.imshow(r)
            plt.title("segmentation")
            plt.savefig("road_segmentation_result_%03.0f.png"%fig_ctr)
            plt.close()
            fig_ctr += 1
            if fig_ctr > n_images:
                break

def main():
    # only download dataset once. This takes a while.
    # heuristic that I included to make sure the dataset is only donwloaded and prepared once
    if not os.path.isfile("target_test.npy"):
        prepare_dataset()

    # set some hyper parameters. You should not have to touch anything if you have 4GB or more VRAM
    BATCH_SIZE = 8 # this works if you have ~ 8GB VRAM. Use smaller BATCH_SIZE for other GPUs
    N_EPOCHS = 50
    N_BATCHES_PER_EPOCH = 100
    PATCH_SIZE = 512

    # load the prepared data. They have been converted to np arrays because they are much faster to load than single image files.
    # This code will not load the entire array into memory but rather read from disk the bits that we currently need
    # set mmap_mode to None if you want to load the data into RAM
    mmap_mode = 'r'
    data_train = np.load("data_train.npy", mmap_mode=mmap_mode)
    target_train = np.load("target_train.npy", mmap_mode=mmap_mode)
    data_valid = np.load("data_valid.npy", mmap_mode=mmap_mode)
    target_valid = np.load("target_valid.npy", mmap_mode=mmap_mode)
    data_test = np.load("data_test.npy", mmap_mode=mmap_mode)
    target_test = np.load("target_test.npy", mmap_mode=mmap_mode)

    # we are using pad='same' for simplicity (otherwise we would have to crop our ground truth). Keep in mind that this
    # may not be ideal
    net = build_UNet(n_input_channels=3, BATCH_SIZE=None, num_output_classes=2, pad='same',
                     nonlinearity=lasagne.nonlinearities.rectify, input_dim=(PATCH_SIZE, PATCH_SIZE),
                     base_n_filters=16, do_dropout=False)
    output_layer_for_loss = net["output_flattened"]

    # this is np.sum(target_train == 0) and np.sum(target_train == 1). No need to compute this every time
    class_frequencies = np.array([2374093357., 118906643.])
    # we will reweight the loss to put more focus on road pixels (because of class imbalance). This is a simple approach
    # and could be improved if you also have a class imbalance in your experiments.
    # we are taking **0.25 here because we want the net to focus more on the road pixels but not too much (otherwise
    # it would not be penalized enough for missclassifying terrain pixels which results in too many false positives)
    class_weights = (class_frequencies[[1,0]])**0.25
    class_weights = class_weights / np.sum(class_weights) * 2.
    class_weights = class_weights.astype(np.float32)

    # if you wish to load pretrained weights you can uncomment this code
    # val accuracy:  0.966384  val loss:  0.0947428  val AUC score:  0.980004909707
    # you can also change the lower part of this code to load your own pretrained params
    '''if not os.path.isfile('UNet_params_pretrained.pkl'):
        import urllib
        import zipfile
        urllib.urlretrieve("https://s3.amazonaws.com/lasagne/recipes/pretrained/UNet_mass_road_segm_params.zip", 'pretrained_weights.zip')
        zip_ref = zipfile.ZipFile('pretrained_weights.zip', 'r')
        zip_ref.extractall("./")
        zip_ref.close()
    with open("UNet_params_pretrained.pkl", 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer_for_loss, params)'''

    x_sym = T.tensor4()
    seg_sym = T.ivector()
    w_sym = T.vector()

    # add some weight decay
    l2_loss = lasagne.regularization.regularize_network_params(output_layer_for_loss, lasagne.regularization.l2) * 1e-4

    # the distinction between prediction_train and test is important only if we enable dropout
    prediction_train = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=False, batch_norm_update_averages=False, batch_norm_use_averages=False)
    # we could use a binary loss but I stuck with categorical crossentropy so that less code has to be changed if your
    # application has more than two classes
    loss = lasagne.objectives.categorical_crossentropy(prediction_train, seg_sym)
    loss *= w_sym
    loss = loss.mean()
    loss += l2_loss
    acc_train = T.mean(T.eq(T.argmax(prediction_train, axis=1), seg_sym), dtype=theano.config.floatX)

    prediction_test = lasagne.layers.get_output(output_layer_for_loss, x_sym, deterministic=True, batch_norm_update_averages=False, batch_norm_use_averages=False)
    loss_val = lasagne.objectives.categorical_crossentropy(prediction_test, seg_sym)

    # we multiply our loss by a weight map. In this example the weight map simply increases the loss for road pixels and
    # decreases the loss for other pixels. We do this to ensure that the network puts more focus on getting the roads
    # right
    loss_val *= w_sym
    loss_val = loss_val.mean()
    loss_val += l2_loss
    acc = T.mean(T.eq(T.argmax(prediction_test, axis=1), seg_sym), dtype=theano.config.floatX)

    # learning rate has to be a shared variable because we decrease it with every epoch
    params = lasagne.layers.get_all_params(output_layer_for_loss, trainable=True)
    learning_rate = theano.shared(np.float32(0.001))
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

    # create a convenience function to get the segmentation

    seg_output = lasagne.layers.get_output(net["output_segmentation"], x_sym, deterministic=True, batch_norm_update_averages=False, batch_norm_use_averages=False)
    seg_output = seg_output.argmax(1)

    train_fn = theano.function([x_sym, seg_sym, w_sym], [loss, acc_train], updates=updates)
    val_fn = theano.function([x_sym, seg_sym, w_sym], [loss_val, acc])
    get_segmentation = theano.function([x_sym], seg_output)
    # we need this for calculating the AUC score
    get_class_probas = theano.function([x_sym], prediction_test)

    # some data augmentation. If you want better results you should invest more effort here. I left rotations and
    # deformations out for the sake of speed and simplicity
    train_generator = random_crop_generator(batch_generator(data_train, target_train, BATCH_SIZE, shuffle=True), PATCH_SIZE)
    train_generator = threaded_generator(train_generator, num_cached=10)

    # do the actual training
    for epoch in np.arange(0, N_EPOCHS):
        print epoch
        losses_train = []
        n_batches = 0
        accuracies_train = []
        for data, target in train_generator:
            # the output of the net has shape (BATCH_SIZE, N_CLASSES). We therefore need to flatten the segmentation so
            # that we can match it with the prediction via the crossentropy loss function
            target_flat = target.ravel()
            loss, acc = train_fn(data.astype(np.float32), target_flat, class_weights[target_flat])
            losses_train.append(loss)
            accuracies_train.append(acc)
            n_batches += 1
            if n_batches > N_BATCHES_PER_EPOCH:
                break
        print "epoch: ", epoch, "\ntrain accuracy: ", np.mean(accuracies_train), " train loss: ", np.mean(losses_train)

        losses_val = []
        accuracies_val = []
        auc_val = []
        # there is no need for data augmentation on the validation. However we need patches of the same size which is why
        # we are using center crop generator
        # since the validation generator does not loop around we need to reinstantiate it for every epoch
        validation_generator = center_crop_generator(batch_generator(data_valid, target_valid, BATCH_SIZE, shuffle=False), PATCH_SIZE)
        validation_generator = threaded_generator(validation_generator, num_cached=10)
        for data, target in validation_generator:
            target_flat = target.ravel()
            loss, acc = val_fn(data.astype(np.float32), target_flat, class_weights[target_flat])
            losses_val.append(loss)
            accuracies_val.append(acc)
            auc_val.append(roc_auc_score(target_flat, get_class_probas(data)[:, 1]))
        print "val accuracy: ", np.mean(accuracies_val), " val loss: ", np.mean(losses_val), " val AUC score: ", np.mean(auc_val)
        learning_rate *= 0.8
        # save trained weights after each epoch
        with open("UNet_params_ep%03.0f.pkl"%epoch, 'w') as f:
            cPickle.dump(lasagne.layers.get_all_param_values(output_layer_for_loss), f)

    # create some png files showing (raw image, ground truth, prediction). Of course we use the test set here ;-)
    test_gen = random_crop_generator(batch_generator(data_test, target_test, BATCH_SIZE), PATCH_SIZE)
    plot_some_results(get_segmentation, test_gen, 15)


if __name__ == "__main__":
    main()
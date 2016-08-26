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

def prep_folders():
    if not os.path.isdir("data"):
        os.mkdir("data")

    if not os.path.isdir("data/validation"):
        os.mkdir("data/validation")
    if not os.path.isdir("data/training"):
        os.mkdir("data/training")
    if not os.path.isdir("data/test"):
        os.mkdir("data/test")

    if not os.path.isdir("data/validation/sat_img"):
        os.mkdir("data/validation/sat_img")
    if not os.path.isdir("data/validation/map"):
        os.mkdir("data/validation/map")
    if not os.path.isdir("data/training/sat_img"):
        os.mkdir("data/training/sat_img")
    if not os.path.isdir("data/training/map"):
        os.mkdir("data/training/map")
    if not os.path.isdir("data/test/sat_img"):
        os.mkdir("data/test/sat_img")
    if not os.path.isdir("data/test/map"):
        os.mkdir("data/test/map")

def prep_urls():
    valid_data_url = valid_target_url = np.loadtxt("mass_roads_validation.txt", dtype=str)
    valid_data_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/"
    valid_target_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/"

    train_data_url = train_target_url  = np.loadtxt("mass_roads_train.txt", dtype=str)
    train_data_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/"
    train_target_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/"

    test_data_url = test_target_url  = np.loadtxt("mass_roads_test.txt", dtype=str)
    test_data_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/"
    test_target_str = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/map/"

    f = open("mass_roads_train_data_download.sh", 'w')
    g = open("mass_roads_train_target_download.sh", 'w')
    for img_name in train_data_url:
        f.write("wget -O data/training/sat_img/%sf "%img_name + train_data_str + img_name + "f" + "\n")
        g.write("wget -O data/training/map/%s "%img_name + train_target_str + img_name + "\n")
    f.close()
    g.close()

    f = open("mass_roads_validation_data_download.sh", 'w')
    g = open("mass_roads_validation_target_download.sh", 'w')
    for img_name in valid_data_url:
        f.write("wget -O data/validation/sat_img/%s "%img_name + valid_data_str + img_name + "\n")
        g.write("wget -O data/validation/map/%s "%img_name[:-1] + valid_target_str + img_name[:-1] + "\n")
    f.close()
    g.close()

    f = open("mass_roads_test_data_download.sh", 'w')
    g = open("mass_roads_test_target_download.sh", 'w')
    for img_name in test_data_url:
        f.write("wget -O data/test/sat_img/%s "%img_name + test_data_str + img_name + "\n")
        g.write("wget -O data/test/map/%s "%img_name[:-1] + test_target_str + img_name[:-1] + "\n")
    f.close()
    g.close()

def download_dataset():
    os.system("sh mass_roads_train_data_download.sh &")
    os.system("sh mass_roads_train_target_download.sh &")
    os.system("sh mass_roads_validation_data_download.sh &")
    os.system("sh mass_roads_validation_target_download.sh &")
    os.system("sh mass_roads_test_data_download.sh &")
    os.system("sh mass_roads_test_target_download.sh &")

def load_data(folder):
    images_sat = [img for img in os.listdir(os.path.join(folder, "sat_img")) if fnmatch.fnmatch(img, "*.tif*")]
    images_map = [img for img in os.listdir(os.path.join(folder, "map")) if fnmatch.fnmatch(img, "*.tif*")]
    assert(len(images_sat) == len(images_map))
    images_sat.sort()
    images_map.sort()
    # images are 1500 by 1500 pixels each
    data = np.zeros((len(images_sat), 3, 1500, 1500), dtype=np.uint8)
    target = np.zeros((len(images_sat), 1, 1500, 1500), dtype=np.uint8)
    ctr = 0
    for sat_im, map_im in zip(images_sat, images_map):
        data[ctr] = plt.imread(os.path.join(folder, "sat_img", sat_im)).transpose((2, 0, 1))
        # target has values 0 and 255. make that 0 and 1
        target[ctr, 0] = plt.imread(os.path.join(folder, "map", map_im))/255
        ctr += 1
    return data, target

def batch_generator(data, target, BATCH_SIZE):
    '''
    just a simple batch iterator, no cropping, no rotation, no anything
    '''
    np.random.seed()
    idx = np.arange(data.shape[0])
    while True:
        ids = np.random.choice(idx, BATCH_SIZE)
        yield np.array(data[ids]), np.array(target[ids])

def random_crop_generator(generator, crop_size=(128, 128)):
    '''
    yields a random crop of size crop_size
    '''
    np.random.seed()
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size, crop_size]
    elif len(crop_size) == 2:
        crop_size = list(crop_size)
    else:
        raise ValueError("invalid crop_size")
    for data, seg in generator:
        lb_x = np.random.randint(0, data.shape[2]-crop_size[0])
        lb_y = np.random.randint(0, data.shape[3]-crop_size[1])
        data = data[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1]]
        seg = seg[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1]]
        yield data, seg

def threaded_generator(generator, num_cached=10):
    # this code is written by jan Schluter
    # copied from https://github.com/benanne/Lasagne/issues/12
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
        queue.task_done()
        item = queue.get()

def prepare_dataset():
    prep_folders()
    prep_urls()
    download_dataset()
    # the dataset is now downloaded in the background. Once every few seconds we check if the download is done. We do
    # this by checking whether tha last training image exists
    while not os.path.isfile("data/training/map/99238675_15.tiff") and not os.path.isfile("data/training/sat_img/99238675_15.tiff"):
        print "download seems to be running..."
        sleep(5)
    print "download done..."
    try:
        data_train, target_train = load_data("data/training")
        data_valid, target_valid = load_data("data/validation")
        data_test, target_test = load_data("data/test")
        # loading np arrays is much faster than loading the images one by one every time
        np.save("train_data.npy", data_train)
        np.save("train_target.npy", target_train)
        np.save("valid_data.npy", data_valid)
        np.save("valid_target.npy", target_valid)
        np.save("test_data.npy", data_test)
        np.save("test_target.npy", target_test)
    except:
        print "something went wrong, maybe the download?"


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
    # TODO is it worth the effort to refactor everything so that we use memmaps?
    data_train = np.load("train_data.npy")
    target_train = np.load("train_target.npy")
    data_valid = np.load("valid_data.npy")
    target_valid= np.load("valid_target.npy")
    data_test = np.load("test_data.npy")
    target_test = np.load("test_target.npy")

    # we are using pad='same' for simplicity (otherwise we would have to crop our ground truth).
    # Did not test for other paddings TODO
    net = build_UNet(n_input_channels=3, BATCH_SIZE=BATCH_SIZE, num_output_classes=2, pad='same',
                     nonlinearity=lasagne.nonlinearities.elu, input_dim=(PATCH_SIZE, PATCH_SIZE),
                     base_n_filters=16, do_dropout=False)
    output_layer_for_loss = net["output_flattened"]

    # if you wish to load pretrained weights you can uncomment this code and modify the file name
    # if you want, use my pretained weights (got around 96% accuracy and a loss of 0.11 using excessive data
    # augmentation)
    # https://www.dropbox.com/s/t6juf6o2ix7dntk/UNet_roadSegmentation_Params.zip?dl=0
    '''with open("UNet_params_ep0.pkl", 'r') as f:
        params = cPickle.load(f)
        lasagne.layers.set_all_param_values(output_layer, params)'''

    x_sym = T.tensor4()
    seg_sym = T.ivector()
    w_sym = T.vector()

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

    train_fn = theano.function([x_sym, seg_sym, w_sym], [loss, acc_train], updates=updates)
    val_fn = theano.function([x_sym, seg_sym, w_sym], [loss_val, acc])
    get_segmentation = theano.function([x_sym], seg_output)


    # this is np.sum(target_train == 0) and np.sum(target_train == 1). No need to compute this every time
    class_frequencies = np.array([2374093357., 118906643.])
    # we are taking the log here because we want the net to focus more on the road pixels but not too much (otherwise
    # it would not be penalized enough for missclassifying terrain pixels which results in too many false positives)
    class_weights = np.log(class_frequencies[[1,0]])
    class_weights = class_weights / np.sum(class_weights) * 2.
    class_weights = class_weights.astype(np.float32)

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
            loss, acc = train_fn(data.astype(np.float32), target_flat, class_weights[target_flat])
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
            loss, acc = val_fn(data.astype(np.float32), target_flat, class_weights[target_flat])
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
# coding=utf-8

import numpy as np
import lasagne as nn
from lasagne.layers import dnn
from functools import partial
import os
import utils
import pickle

conv3 = partial(dnn.Conv2DDNNLayer,
                stride=(1, 1),
                border_mode="same",
                filter_size=(3, 3),
                nonlinearity=nn.nonlinearities.rectify)

dense = partial(nn.layers.DenseLayer,
                nonlinearity=nn.nonlinearities.rectify)

max_pool = partial(dnn.MaxPool2DDNNLayer,
                   pool_size=(2, 2),
                   stride=(2, 2))


def vgg16(batch_shape):
    """
    Create a vgg16, with the parameters from http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    See googlenet.py for the method used to convert these caffe parameters to lasagne parameters.
    :param batch_shape: The shape of the input images. This should be of size (N, 3, X>=224, Y>=224). Note flexible
    image size, as the last dense layers have been implemented here with convolutional layers.
    :return: a struct with the input layer, the logit layer (before the final softmax) and the output layer.
    """
    l_in = nn.layers.InputLayer(shape=batch_shape)
    l = l_in

    l = conv3(l, num_filters=64)
    l = conv3(l, num_filters=64)

    l = max_pool(l)

    l = conv3(l, num_filters=128)
    l = conv3(l, num_filters=128)

    l = max_pool(l)

    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)
    l = conv3(l, num_filters=256)

    l = max_pool(l)

    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)

    l = max_pool(l)

    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)
    l = conv3(l, num_filters=512)

    l = max_pool(l)

    l = dnn.Conv2DDNNLayer(l,
                           num_filters=4096,
                           stride=(1, 1),
                           border_mode="valid",
                           filter_size=(7, 7))
    l = dnn.Conv2DDNNLayer(l,
                           num_filters=4096,
                           stride=(1, 1),
                           border_mode="same",
                           filter_size=(1, 1))

    l_logit = dnn.Conv2DDNNLayer(l,
                                 num_filters=1000,
                                 stride=(1, 1),
                                 border_mode="same",
                                 filter_size=(1, 1),
                                 nonlinearity=None)

    l_logit_flat = nn.layers.FlattenLayer(l_logit)
    l_dense = nn.layers.NonlinearityLayer(l_logit_flat, nonlinearity=nn.nonlinearities.softmax)

    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vgg16.pkl')

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            nn.layers.set_all_param_values(l_dense, pickle.load(f))

    return utils.struct(
        input=l_in,
        logit=l_logit,
        out=l_dense
    )


if __name__ == "__main__":
    model = vgg16((1, 3, 224, 224))
    nn.layers.set_all_param_values(model.out, np.load("data/vgg16.npy"))
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vgg16.pkl')
    pickle.dump(nn.layers.get_all_param_values(model.out), open(filename, 'w'))

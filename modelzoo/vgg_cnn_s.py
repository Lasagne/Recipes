# VGG_CNN_S, model from the paper:
# "Return of the Devil in the Details: Delving Deep into Convolutional Nets"
# 13.1% top-5 error on ILSVRC-2012-val
# Original source: https://gist.github.com/ksimonyan/fd8800eeb36e276cd6f9
# License: non-commercial use only

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg_cnn_s.pkl

from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.nonlinearities import softmax


def build_model():
    net = {}

    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'],
                             num_filters=96,
                             filter_size=7,
                             stride=2)
    # caffe has alpha = alpha * pool_size
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001)
    net['pool1'] = PoolLayer(net['norm1'],
                             pool_size=3,
                             stride=3,
                             ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
    net['pool2'] = PoolLayer(net['conv2'],
                             pool_size=2,
                             stride=2,
                             ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'],
                             num_filters=512,
                             filter_size=3,
                             pad=1)
    net['conv4'] = ConvLayer(net['conv3'],
                             num_filters=512,
                             filter_size=3,
                             pad=1)
    net['conv5'] = ConvLayer(net['conv4'],
                             num_filters=512,
                             filter_size=3,
                             pad=1)
    net['pool5'] = PoolLayer(net['conv5'],
                             pool_size=3,
                             stride=3,
                             ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

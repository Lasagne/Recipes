__author__ = 'Guido Zuidhof'
import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, ConcatLayer, Upscale2DLayer
from lasagne.init import HeNormal
from lasagne import nonlinearities
from lasagne.regularization import l2, regularize_network_params

# Unet implementation, based on the architecture proposed in 
#    U-Net: Convolutional Networks for Biomedical Image Segmentation  
#    Olaf Ronneberger, Philipp Fischer, Thomas Brox
#    Medical Image Computing and Computer-Assisted Intervention (MICCAI), Springer, LNCS, Vol.9351: 234--241, 2015, 
#    available at arXiv:1505.04597 [cs.CV] 
#
# (see http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

# The output of the network is a segmentation of the input image. This is a crop of the 
# original image as no padding is applied (because this may introduce a border effect).
# To determine the size of this output image, you can use the function output_size_for_input (default is 388x388 pixels).

# The input is an image, the segmentation truth image, and a weight map. This can be used to make certain areas
# in the image more important than others when determining the loss.


# Utility function, for determining the output size offline
def output_size_for_input(in_size, depth):
    in_size = np.array(in_size)
    in_size -= 4
    for _ in range(depth-1):
        in_size = in_size//2
        in_size -= 4
    for _ in range(depth-1):
        in_size = in_size*2
        in_size -= 4
    return in_size

def _num_filters_for_depth(depth, branching_factor):
    return 2**(branching_factor+depth)

def define_network(input_var, input_size=(572,572), 
                        depth=5, 
                        branching_factor=6, #2^6 filters for first level, 2^7 for second, etc.
                        num_input_channels=1,
                        num_classes=2):
    batch_size = None
    nonlinearity = nonlinearities.rectify


    net = {}
    net['input'] = InputLayer(shape=(batch_size, num_input_channels, input_size[0],input_size[1]), input_var=input_var)
    def contraction(depth, deepest):
        n_filters = _num_filters_for_depth(depth, branching_factor)
        incoming = net['input'] if depth == 0 else net['pool{}'.format(depth-1)]

        net['conv{}_1'.format(depth)] = Conv2DLayer(incoming,
                                    num_filters=n_filters, filter_size=3, pad='valid',
                                    W=HeNormal(gain='relu'),
                                    nonlinearity=nonlinearity)

        net['conv{}_2'.format(depth)] = Conv2DLayer(net['conv{}_1'.format(depth)],
                                    num_filters=n_filters, filter_size=3, pad='valid', 
                                    W=HeNormal(gain='relu'),
                                    nonlinearity=nonlinearity)

        if not deepest:
            net['pool{}'.format(depth)] = MaxPool2DLayer(net['conv{}_2'.format(depth)], pool_size=2, stride=2)

    def expansion(depth, deepest):
        n_filters = _num_filters_for_depth(depth, branching_factor)

        incoming = net['conv{}_2'.format(depth+1)] if deepest else net['_conv{}_2'.format(depth+1)]

        upscaling = Upscale2DLayer(incoming, 4)
        net['upconv{}'.format(depth)] = Conv2DLayer(upscaling,
                                        num_filters=n_filters, filter_size=2, stride=2,
                                        W=HeNormal(gain='relu'),
                                        nonlinearity=nonlinearity)

        net['bridge{}'.format(depth)] = ConcatLayer([
                                        net['upconv{}'.format(depth)],
                                        net['conv{}_2'.format(depth)]],
                                        axis=1, cropping=[None, None, 'center', 'center'])

        net['_conv{}_1'.format(depth)] = Conv2DLayer(net['bridge{}'.format(depth)],
                                        num_filters=n_filters, filter_size=3, pad='valid',
                                        W=HeNormal(gain='relu'),
                                        nonlinearity=nonlinearity)

        net['_conv{}_2'.format(depth)] = Conv2DLayer(net['_conv{}_1'.format(depth)],
                                        num_filters=n_filters, filter_size=3, pad='valid',
                                        W=HeNormal(gain='relu'),
                                        nonlinearity=nonlinearity)

    # Contraction
    for d in range(depth):
        #There is no pooling at the last layer
        deepest = d == depth-1
        contraction(d, deepest)

    # Expansion
    for d in reversed(range(depth-1)):
        deepest = d == depth-2
        expansion(d, deepest)

    # Output layer
    net['out'] = Conv2DLayer(net['_conv0_2'], num_filters=num_classes, filter_size=(1,1), pad='valid',
                                    nonlinearity=None)

    print ('Network output shape '+ str(lasagne.layers.get_output_shape(net['out'])))
    return net


def define_updates(network, input_var, target_var, weight_var, learning_rate=0.01, momentum=0.9, l2_lambda=1e-5):
    params = lasagne.layers.get_all_params(network, trainable=True)

    out = lasagne.layers.get_output(network)
    test_out = lasagne.layers.get_output(network, deterministic=True)

    l2_loss = l2_lambda * regularize_network_params(network, l2)

    train_metrics = _score_metrics(out, target_var, weight_var, l2_loss)
    loss, acc, target_prediction, prediction = train_metrics

    val_metrics = _score_metrics(test_out, target_var, weight_var, l2_loss)
    t_loss, t_acc, t_target_prediction, t_prediction = val_metrics


    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=learning_rate, momentum=momentum)

    train_fn = theano.function([input_var, target_var, weight_var],[
                                loss, l2_loss, acc, target_prediction, prediction],
                                updates=updates)

    val_fn = theano.function([input_var, target_var, weight_var], [
                                t_loss, l2_loss, t_acc, t_target_prediction, t_prediction])


    return train_fn, val_fn

def define_predict(network, input_var):
    params = lasagne.layers.get_all_params(network, trainable=True)
    out = lasagne.layers.get_output(network, deterministic=True)
    out_flat = out.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    prediction = lasagne.nonlinearities.softmax(out_flat)

    print "Defining predict"
    predict_fn = theano.function([input_var],[prediction])

    return predict_fn

def _score_metrics(out, target_var, weight_map, l2_loss=0):
    _EPSILON=1e-8

    out_flat = out.dimshuffle(1,0,2,3).flatten(ndim=2).dimshuffle(1,0)
    target_flat = target_var.dimshuffle(1,0,2,3).flatten(ndim=1)
    weight_flat = weight_map.dimshuffle(1,0,2,3).flatten(ndim=1)

    # Softmax output, original paper may have used a sigmoid output_size_for_input
    # but here we opt for softmax, as this also works for multiclass segmentation.
    prediction = lasagne.nonlinearities.softmax(out_flat)

    loss = lasagne.objectives.categorical_crossentropy(T.clip(prediction,_EPSILON,1-_EPSILON), target_flat)
    loss = loss * weight_flat
    loss = loss.mean()
    loss += l2_loss

    # Pixelwise accuracy
    accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), target_flat),
                      dtype=theano.config.floatX)

    return loss, accuracy, target_flat, prediction
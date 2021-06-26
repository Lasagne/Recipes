# C3D, AlexNet type model with 3D convolutions (for video processing).
# From "Learning Spatiotemporal Features with 3D Convolutional Networks"
#
# Pretrained weights from https://s3.amazonaws.com/lasagne/recipes/pretrained/c3d/c3d_model.pkl
# and the snipplet mean from
# https://s3.amazonaws.com/lasagne/recipes/pretrained/c3d/snipplet_mean.npy
#
# License: Not specified
# Author: Michael Gygli, https://github.com/gyglim
#
import lasagne
from lasagne.layers.shape import PadLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.dnn import Conv3DDNNLayer, MaxPool3DDNNLayer
from lasagne.nonlinearities import softmax

import theano
import numpy as np
import skimage.transform
from skimage import color
import pickle

def build_model():
    '''
    Builds C3D model

    Returns
    -------
    dict
        A dictionary containing the network layers, where the output layer is at key 'prob'
    '''
    net = {}
    net['input'] = InputLayer((None, 3, 16, 112, 112))

    # ----------- 1st layer group ---------------
    net['conv1a'] = Conv3DDNNLayer(net['input'], 64, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify,flip_filters=False)
    net['pool1']  = MaxPool3DDNNLayer(net['conv1a'],pool_size=(1,2,2),stride=(1,2,2))

    # ------------- 2nd layer group --------------
    net['conv2a'] = Conv3DDNNLayer(net['pool1'], 128, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool2']  = MaxPool3DDNNLayer(net['conv2a'],pool_size=(2,2,2),stride=(2,2,2))

    # ----------------- 3rd layer group --------------
    net['conv3a'] = Conv3DDNNLayer(net['pool2'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['conv3b'] = Conv3DDNNLayer(net['conv3a'], 256, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool3']  = MaxPool3DDNNLayer(net['conv3b'],pool_size=(2,2,2),stride=(2,2,2))

    # ----------------- 4th layer group --------------
    net['conv4a'] = Conv3DDNNLayer(net['pool3'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['conv4b'] = Conv3DDNNLayer(net['conv4a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['pool4']  = MaxPool3DDNNLayer(net['conv4b'],pool_size=(2,2,2),stride=(2,2,2))

    # ----------------- 5th layer group --------------
    net['conv5a'] = Conv3DDNNLayer(net['pool4'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    net['conv5b'] = Conv3DDNNLayer(net['conv5a'], 512, (3,3,3), pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    # We need a padding layer, as C3D only pads on the right, which cannot be done with a theano pooling layer
    net['pad']    = PadLayer(net['conv5b'],width=[(0,1),(0,1)], batch_ndim=3)
    net['pool5']  = MaxPool3DDNNLayer(net['pad'],pool_size=(2,2,2),pad=(0,0,0),stride=(2,2,2))
    net['fc6-1']  = DenseLayer(net['pool5'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    net['fc7-1']  = DenseLayer(net['fc6-1'], num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    net['fc8-1']  = DenseLayer(net['fc7-1'], num_units=487, nonlinearity=None)
    net['prob']  = NonlinearityLayer(net['fc8-1'], softmax)

    return net

def set_weights(net,model_file):
    '''
    Sets the parameters of the model using the weights stored in model_file
    Parameters
    ----------
    net: a Lasagne layer

    model_file: string
        path to the model that containes the weights

    Returns
    -------
    None

    '''
    with open(model_file) as f:
        print('Load pretrained weights from %s...' % model_file)
        model = pickle.load(f)
    print('Set the weights...')
    lasagne.layers.set_all_param_values(net, model,trainable=True)


######## Below, there are several helper functions to transform (lists of) images into the right format  ######

def get_snips(images,image_mean,start=0, with_mirrored=False):
    '''
    Converts a list of images to a 5d tensor that serves as input to C3D
    Parameters
    ----------
    images: 4d numpy array or list of 3d numpy arrays
        RGB images

    image_mean: 4d numpy array
        snipplet mean (given by C3D)

    start: int
        first frame to use from the list of images

    with_mirrored: bool
        return the snipplet and its mirrored version (horizontal flip)

    Returns
    -------
    caffe format 5D numpy array (serves as input to C3D)

    '''
    assert len(images) >= start+16, "Not enough frames to fill a snipplet of 16 frames"

    # Convert images to caffe format and stack them
    caffe_imgs=map(lambda x: rgb2caffe(x).reshape(1,3,128,171),images[start:start+16])
    snip=np.vstack(caffe_imgs).swapaxes(0,1)

    # Remove the mean
    snip-= image_mean

    # Get the center crop
    snip=snip[:,:,8:120,29:141]
    snip=snip.reshape(1,3,16,112,112)

    if with_mirrored: # Return nromal and flipped version
        return np.vstack((snip,snip[:,:,:,:,::-1]))
    else:
        return snip


def rgb2caffe(im, out_size=(128, 171)):
    '''
    Converts an RGB image to caffe format and downscales it as needed by C3D

    Parameters
    ----------
    im numpy array
        an RGB image
    downscale

    Returns
    -------
    a caffe image (channel,height, width) in BGR format

    '''
    im=np.copy(im)
    if len(im.shape)==2: # Make sure the image has 3 channels
        im = color.gray2rgb(im)

    h, w, _ = im.shape
    im = skimage.transform.resize(im, out_size,)
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    return np.array(im,theano.config.floatX)


def convert_back(raw_im, image_mean=None,idx=0):
    '''
    Converts a Caffe format image back to the standard format, so that it can be plotted.

    Parameters
    ----------
    raw_im numpy array
        a bgr caffe image; format (channel,height, width)
    add_mean boolean
        Add the C3D mean?
    idx integer (default: 0)
        position in the snipplet (used for mean addtion, but differences are very small)

    Returns
    -------
    a RGB image; format (w,h,channel)
    '''

    raw_im=np.copy(raw_im)
    if image_mean is not None:
        raw_im += image_mean[idx,:,8:120,29:141].squeeze()

    # Convert to RGB
    raw_im = raw_im[::-1, :, :]

    # Back in (y,w,channel) order
    im = np.array(np.swapaxes(np.swapaxes(raw_im, 1, 0), 2, 1),np.uint8)
    return im

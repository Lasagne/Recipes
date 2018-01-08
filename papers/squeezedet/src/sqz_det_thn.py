"""
Copyright (c) 2017 Corvidim (corvidim.net)
Licensed under the BSD 2-Clause License (see LICENSE for details)
Authors: V. Ablavsky, A. J. Fox
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import argparse, os, sys, gzip
from pdb import set_trace as keyboard

import utils

import numpy as np
import PIL.Image

python_ver =  sys.version_info[0]
import pickle

import lasagne
import theano, theano.tensor as T
import collections

import kitti_squeezeDet_config

import matplotlib
# Python 2 vs. 3 backends
if sys.platform=='darwin' and sys.version_info[0] == 3:
    matplotlib.use('qt5agg')
# if headless, we need this *before* importing plt
if utils.is_headless():
    orig_backend = matplotlib.rcParams['backend']
    matplotlib.use('Agg')
    print('Headless; resetting backend from {} to {}'
          ' and assuming no_gui'.format(orig_backend, matplotlib.rcParams['backend']))
import matplotlib.pyplot as plt

######################################################################################
#                  load_network_weights()
######################################################################################
def load_network_weights(par):
    print('********* load_network_weights()')
    try:
        path_ = par['network_weights']
        dir_, file_ = os.path.split(path_)
        base_file, base_ext = os.path.splitext(file_)
        assert base_ext == '.gz' and os.path.splitext(base_file)[1] == '.pkl'

        with gzip.open(path_, 'rb') as ar:
            # This .pkl was produced with Py2K; if loading in Py3K we need to specify encoding
            if python_ver < 3:
                network_weights = pickle.load(ar)
            else:
                network_weights = pickle.load(ar, encoding='latin-1')
    except:
        raise ValueError('Expected --network_weights=/PATH/TO/FILE.pkl.gz')
    return network_weights

######################################################################################
#                  viz_det_roi
######################################################################################
def viz_det_roi(img, det_roi, det_label, par, out_file_name, plot_title=''):
    print('********* viz_det_roi()')
    """
    img: RGB order, normalized to [0,1]
    """

    if utils.is_headless():
        par['no_gui'] = True

    cls2clr = {
        'car': (0, 0.75, 1),
        'cyclist': (1, 0.75, 0),
        'ped':(1, 0, 0.75)
    }

    if not par['no_gui']:
        plt.ion()

    plt.figure()
    plt.suptitle('viz_det_roi(): {} bounding box(es)'.format(len(det_roi)))
    plt.title(plot_title)

    plt.imshow(img)
    plt.axis('off')
    ax=plt.gca()

    for bbox, label in zip(det_roi, det_label):
        # bbox of form [cx, cy, w, h]
        w = bbox[2]
        h = bbox[3]
        [xmin, ymin, xmax, ymax] = utils.bbox_transform(bbox)
        class_str = label.split(':')[0]
        color_val = cls2clr[class_str]

        print('adding box: {} [~{}x{}: xmin:{:.2f}, ymin:{:.2f}),'
              ' (xmax:{:.2f}, ymax:{:.2f}] '.format(label,
                                                    int(ymax-ymin),
                                                    int(xmax-xmin),
                                                    xmin,
                                                    ymin,
                                                    xmax,
                                                    ymax))

        rect = matplotlib.patches.Rectangle((xmin,ymin),w,h,edgecolor=color_val,
                                            facecolor='none',linewidth=0.5)
        ax.add_patch(rect)
        ax.text(xmin,ymin+h,label,fontdict=dict(color=color_val,fontsize=6))

    if not par['no_gui']:
        plt.show()

    plt.savefig(out_file_name,bbox_inches='tight',pad_inches=0.0)
    print ('Image detection output saved to {}'.format(out_file_name))

######################################################################################
#                  load_img()
######################################################################################
def load_img(img_path,mean_img_BGR,par):
    print('********* load_img()')
    raw_PIL = PIL.Image.open(img_path)
    img_RGB = np.array(raw_PIL).astype('float32') # shape (375, 1242, 3), pixel values 0..255
    img_BGR = img_RGB[:,:,::-1]                   # RGB -> BGR order, to match sqzdet
    img_ = img_BGR - mean_img_BGR                 # subtract mean image, to match sqzdet

    img_lasagne = np.expand_dims(np.rollaxis(img_,2),axis=0)  # shape (1, 3, 375, 1242)

    if par['verbose']:
        print('incoming image shape: {} ({})'.format(img_RGB.shape, img_path))
    return (img_RGB, img_lasagne)


######################################################################################
#                  make_sqz_det_net()
######################################################################################
def make_sqz_det_net(thn_x, par):
    """
    Input: thn_x a 4-d tensor (batch_idx,channel_idx,row_idx,col_idx)
           par a dictionary of network specification

    Output: Lasagne network as an ordered dictionary
    """
    print('-'*5 + 'make_sqz_det_net()')
    input_shape = (None,3,None,None) # assume 3 channels, but not input image dimensions
    net = collections.OrderedDict()

    net['input'] = lasagne.layers.InputLayer(shape=input_shape, input_var=thn_x)
    net['conv1'] = lasagne.layers.Conv2DLayer(incoming=net['input'], num_filters=64, filter_size=3, stride=2, flip_filters=False, pad='valid')
    net['pool1'] = lasagne.layers.MaxPool2DLayer(incoming=net['conv1'], stride=2, pool_size=3)
    net['conv2_fire1_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['pool1'], num_filters=16, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv3_fire1_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv2_fire1_sqz'], num_filters=64, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv4_fire1_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv2_fire1_sqz'], num_filters=64, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat1'] = lasagne.layers.ConcatLayer((net['conv3_fire1_exp'],net['conv4_fire1_exp']),axis=1)
    net['conv5_fire2_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['concat1'], num_filters=16, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv6_fire2_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv5_fire2_sqz'], num_filters=64, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv7_fire2_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv5_fire2_sqz'], num_filters=64, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat2'] = lasagne.layers.ConcatLayer((net['conv6_fire2_exp'],net['conv7_fire2_exp']),axis=1)
    net['pool2'] = lasagne.layers.MaxPool2DLayer(incoming=net['concat2'], stride=2, pool_size=3)
    net['conv8_fire3_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['pool2'], num_filters=32, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv9_fire3_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv8_fire3_sqz'], num_filters=128, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv10_fire3_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv8_fire3_sqz'], num_filters=128, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat3'] = lasagne.layers.ConcatLayer((net['conv9_fire3_exp'],net['conv10_fire3_exp']),axis=1)
    net['conv11_fire4_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['concat3'], num_filters=32, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv12_fire4_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv11_fire4_sqz'], num_filters=128, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv13_fire4_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv11_fire4_sqz'], num_filters=128, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat4'] = lasagne.layers.ConcatLayer((net['conv12_fire4_exp'],net['conv13_fire4_exp']),axis=1)
    net['pool3'] = lasagne.layers.MaxPool2DLayer(incoming=net['concat4'], stride=2, pool_size=3)
    net['conv14_fire5_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['pool3'], num_filters=48, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv15_fire5_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv14_fire5_sqz'], num_filters=192, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv16_fire5_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv14_fire5_sqz'], num_filters=192, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat5'] = lasagne.layers.ConcatLayer((net['conv15_fire5_exp'],net['conv16_fire5_exp']),axis=1)
    net['conv17_fire6_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['concat5'], num_filters=48, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv18_fire6_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv17_fire6_sqz'], num_filters=192, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv19_fire6_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv17_fire6_sqz'], num_filters=192, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat6'] = lasagne.layers.ConcatLayer((net['conv18_fire6_exp'],net['conv19_fire6_exp']),axis=1)
    net['conv20_fire7_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['concat6'], num_filters=64, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv21_fire7_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv20_fire7_sqz'], num_filters=256, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv22_fire7_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv20_fire7_sqz'], num_filters=256, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat7'] = lasagne.layers.ConcatLayer((net['conv21_fire7_exp'],net['conv22_fire7_exp']),axis=1)
    net['conv23_fire8_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['concat7'], num_filters=64, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv24_fire8_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv23_fire8_sqz'], num_filters=256, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv25_fire8_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv23_fire8_sqz'], num_filters=256, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat8'] = lasagne.layers.ConcatLayer((net['conv24_fire8_exp'],net['conv25_fire8_exp']),axis=1)
    net['conv26_fire9_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['concat8'], num_filters=96, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv27_fire9_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv26_fire9_sqz'], num_filters=384, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv28_fire9_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv26_fire9_sqz'], num_filters=384, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat9'] = lasagne.layers.ConcatLayer((net['conv27_fire9_exp'],net['conv28_fire9_exp']),axis=1)
    net['conv29_fire10_sqz'] = lasagne.layers.Conv2DLayer(incoming=net['concat9'], num_filters=96, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv30_fire10_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv29_fire10_sqz'], num_filters=384, filter_size=1, stride=1, flip_filters=False, pad='same')
    net['conv31_fire10_exp'] = lasagne.layers.Conv2DLayer(incoming=net['conv29_fire10_sqz'], num_filters=384, filter_size=3, stride=1, flip_filters=False, pad='same')
    net['concat10'] = lasagne.layers.ConcatLayer((net['conv30_fire10_exp'],net['conv31_fire10_exp']),axis=1)
    net['conv32'] = lasagne.layers.Conv2DLayer(incoming=net['concat10'], num_filters=72, filter_size=3, stride=1, flip_filters=False, pad='same', nonlinearity=None)

    return net

######################################################################################
#                  run_sqz_det_net()
######################################################################################
def run_sqz_det_net(par):
    """
    Create a Lasagne network corresponding to SqueezeDet; load its weights from a .pkl file;
    run it end-to-end and visualize bounding boxes.
    """
    print('-'*5 + 'run_sqz_det_net()')
    thn_x = T.tensor4('thn_x') # (batch, single-channel,h,w)

    verbose = par['verbose']

    cfg_mc = kitti_squeezeDet_config.kitti_squeezeDet_config()  # SqueezeDet: "Model config for pascal dataset"

    # for convenience/legibility, shorten 'pedestrian' -> 'ped'
    cfg_mc.CLASS_NAMES = ('car', 'ped', 'cyclist')

    net = make_sqz_det_net(thn_x, par)

    network_weights = load_network_weights(par)

    weights_list = []
    param_idx = 0  # at end: 64
    for net_idx,layer_name in enumerate(net):
        if 'conv' in layer_name:
            if verbose:
                print('net_idx:{}\tW:{}    \tb:{}  \t{}'.format(net_idx,network_weights[param_idx].shape,network_weights[param_idx+1].shape,layer_name))
            weights_list.append([network_weights[param_idx],network_weights[param_idx+1]])  # W,b
            param_idx += 2
        else:
            if verbose: print('        {} [skipped] {}'.format(net_idx,layer_name))
    final_layer_name = layer_name

    weights_list = [num for elem in weights_list for num in elem]  # flatten list of lists

    lasagne.layers.set_all_param_values(net[final_layer_name], weights_list)

    # load image(s); we assume they are all the same size (but do not yet bother checking that)
    img_list = par['img_in']
    mean_img_BGR = cfg_mc['BGR_MEANS'].astype(np.float32)
    for (img_idx, img_str) in enumerate(img_list):
        img = img_str.strip()
        (img_RGB, img_lasagne) = load_img(img,mean_img_BGR,par)   # shape of $SQDT_ROOT/data/sample.png (1, 3, 375, 1242)
        if img_idx == 0:
            img_RGB_list = [img_RGB]
            img_lasagne_stack = img_lasagne
        else:
            img_RGB_list = img_RGB_list + [img_RGB]
            img_lasagne_stack = np.concatenate((img_lasagne_stack,img_lasagne))

    # run image(s) through network
    net_out = np.array(lasagne.layers.get_output(net[final_layer_name],
                                                    img_lasagne_stack,
                                                    deterministic=True).eval())

    # visualize bounding boxes, relative to original image
    for (img_idx, img_str) in enumerate(img_list):
        img_shape = img_RGB_list[img_idx].shape
        (det_roi, det_label) = utils.get_det_roi(cfg_mc, img_shape, np.expand_dims(net_out[img_idx],axis=0), par)

        file_name = os.path.split(img_list[img_idx])[1]
        base, ext = os.path.splitext(file_name)
        out_file_name = os.path.join(par['out_dir'], 'out_'.format(img_idx+1)+base+'.pdf')

        viz_det_roi(img_RGB_list[img_idx]/255., det_roi, det_label, par, out_file_name, plot_title='img {} of {} ({})'.format(img_idx+1, len(img_list), file_name))

    if not par['no_gui']:
        print('pausing for examination of visualization(s)...')
        keyboard()


    
######################################################################################
#                  get_valid_modes()
######################################################################################
def get_valid_modes():
    valid_modes = ['run_sqz_det_net']
    return valid_modes



######################################################################################
#                  main()
######################################################################################
def main(par):
    mode=par['mode']
    if par['out_dir'] and not os.path.exists(par['out_dir']):
        raise IOError('Specified --out_dir does not exist')
    if mode == 'run_sqz_det_net':
        try:
            run_sqz_det_net(par)
        except ValueError as e:
            print('ValueError: {}'.format(e))
        except IOError as e:
            print('IOError: {}: {}'.format(e.strerror,e.filename))

######################################################################################
#                  __main__
######################################################################################
if __name__ == '__main__':

    def csv_list(string):
        return string.split(',')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='run_sqz_det_net', choices=get_valid_modes())
    parser.add_argument('--network_weights', required=True, help='e.g., data/squeezedet_kitti.pkl.gz')
    parser.add_argument('--img_in', type=csv_list, default='data/sample.png', help='/PATH/TO/INPUT/IMG or comma-separated list of paths')
    parser.add_argument('--out_dir', default='.', help='dir path for output images')
    parser.add_argument('--no_gui', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    par=vars(parser.parse_args(sys.argv[1:]))

    main(par)

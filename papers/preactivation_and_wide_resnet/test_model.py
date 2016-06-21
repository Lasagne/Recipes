"""
Lasagne implementation of CIFAR-10 examples from "Identity Mappings in Deep Residual Networks" (https://arxiv.org/abs/1603.05027) and "Wide Residual Networks" (https://arxiv.org/abs/1605.07146)
"""
import os
import sys
import gzip
import time
import pickle
import datetime
import random
import numpy as np
import pandas as pd

import theano
from theano import tensor as T

import lasagne
from lasagne.updates import nesterov_momentum, adam
from lasagne.layers import helper

from utils import load_pickle_data_test

variant = sys.argv[1] if len(sys.argv) > 1 else 'normal'
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 18
width = int(sys.argv[3]) if len(sys.argv) > 3 else 1
print 'Using %s ResNet with depth %d and width %d.'%(variant,depth,width)

if variant == 'normal':
    from models import ResNet_FullPreActivation as ResNet
elif variant == 'bottleneck':
    from models import ResNet_BottleNeck_FullPreActivation as ResNet
elif variant == 'wide':
    from models import ResNet_FullPre_Wide as ResNet
else:
    print ('Unsupported model %s' % variant)

BATCHSIZE = 1

'''
Set up all theano functions
'''
X = T.tensor4('X')
Y = T.ivector('y')

# set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
# load model
if width > 1:
    output_layer = ResNet(X, n=depth, k=width)
else:
    output_layer = ResNet(X, n=depth)
output_test = lasagne.layers.get_output(output_layer, deterministic=True)

output_class = T.argmax(output_test, axis=1)

# set up training and prediction functions
predict_proba = theano.function(inputs=[X], outputs=output_test)
predict_class = theano.function(inputs=[X], outputs=output_class)

'''
Load data and make predictions
'''
test_X, test_y = load_pickle_data_test()

# load network weights
f = gzip.open('data/weights/%s%d_resnet.pklz'%(variant,depth), 'rb')
all_params = pickle.load(f)
f.close()
helper.set_all_param_values(output_layer, all_params)

#make predictions
pred_labels = []
for j in range((test_X.shape[0] + BATCHSIZE - 1) // BATCHSIZE):
    sl = slice(j * BATCHSIZE, (j + 1) * BATCHSIZE)
    X_batch = test_X[sl]
    pred_labels.extend(predict_class(X_batch))

pred_labels = np.array(pred_labels)
print pred_labels.shape

'''
Compare differences
'''
same = 0
for i in range(pred_labels.shape[0]):
    if test_y[i] == pred_labels[i]:
        same += 1

print'Accuracy on the testing set, ', (float(same) / float(pred_labels.shape[0]))

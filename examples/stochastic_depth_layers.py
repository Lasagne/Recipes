
# coding: utf-8

# In[55]:

import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.regularization import *
from lasagne.random import get_rng
from lasagne.updates import *
from lasagne.init import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from urllib import urlretrieve
import cPickle as pickle
import gzip
import imp
import os
from time import time


# In[58]:

class BinomialDropLayer(Layer):
    def __init__(self, incoming, nonlinearity=rectify, survival_p=0.5,
                 **kwargs):
        super(BinomialDropLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (identity if nonlinearity is None
                             else nonlinearity)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = 1-survival_p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return self.p*input
        else:
            #mask = self._srng.binomial(n=1, p=(self.p), size=(input.shape[0],),
            #    dtype=input.dtype)
            mask = T.zeros((input.shape[0],)) + self._srng.uniform( (1,), 0, 1)[0]
            mask = mask.dimshuffle(0,'x','x','x')
            return mask*input


# In[59]:

class IfElseDropLayer(Layer):
    def __init__(self, incoming, nonlinearity=rectify, survival_p=0.5,
                 **kwargs):
        super(IfElseDropLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (identity if nonlinearity is None
                             else nonlinearity)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = 1-survival_p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return self.p*input
        else:
            return ifelse(
                T.lt(self._srng.uniform( (1,), 0, 1)[0], self.p),
                input,
                T.zeros(input.shape)
            )


# There is a difference between this residual block method and the one that is defined in [link]. When the number of filters is different to the layer's output shape (or the stride is different), instead of using a convolution to make things compatible, we use an average pooling with a pool size of 1 and a the defined stride, followed by (if necessary) adding extra zero-padded feature maps. This is because this is how the authors in [link] have defined it.

# In[38]:

"""

print('Building model...')
model = nn.Sequential()
------> 3, 32,32
model:add(cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
            :init('weight', nninit.kaiming, {gain = 'relu'})
            :init('bias', nninit.constant, 0))
model:add(cudnn.SpatialBatchNormalization(16))
model:add(cudnn.ReLU(true))
------> 16, 32,32   First Group
for i=1,opt.N do   addResidualDrop(model, nil, 16)   end
------> 32, 16,16   Second Group
addResidualDrop(model, nil, 16, 32, 2)
for i=1,opt.N-1 do   addResidualDrop(model, nil, 32)   end
------> 64, 8,8     Third Group
addResidualDrop(model, nil, 32, 64, 2)
for i=1,opt.N-1 do   addResidualDrop(model, nil, 64)   end
------> 10, 8,8     Pooling, Linear, Softmax
model:add(nn.SpatialAveragePooling(8,8)):add(nn.Reshape(64))
if opt.dataset == 'cifar10' or opt.dataset == 'svhn' then
  model:add(nn.Linear(64, 10))
elseif opt.dataset == 'cifar100' then
  model:add(nn.Linear(64, 100))
else
  print('Invalid argument for dataset!')
end


"""


# In[60]:

def residual_block(layer, num_filters, filter_size=3, stride=1, num_layers=2, survival_p=0.5):
    #print "input =", layer.output_shape
    conv = layer
    if (num_filters != layer.output_shape[1]) or (stride != 1):
        layer = Pool2DLayer(layer, pool_size=1, stride=stride, mode="average_inc_pad")
        diff = num_filters-layer.output_shape[1]
        if diff % 2 == 0: 
            width_tp = ((diff/2, diff/2),)
        else:
            width_tp = (((diff/2)+1, diff/2),)
        layer = pad(
            layer, 
            batch_ndim=1, 
            width=width_tp
        )
        #print "layer =", layer.output_shape
    for _ in range(num_layers):
        conv = Conv2DLayer(conv, num_filters, filter_size, stride=stride, pad='same')
        #print "conv =", conv.output_shape
        stride = 1
    nonlinearity = conv.nonlinearity
    conv.nonlinearity = lasagne.nonlinearities.identity
    conv = BinomialDropLayer(conv, survival_p=survival_p)
    return NonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity)


# In[63]:

# architecture from:
# https://github.com/yueatsprograms/Stochastic_Depth/blob/master/main.lua
survival_p = 0.5
layer = InputLayer( (None, 3, 32, 32) )
layer = Conv2DLayer(layer, num_filters=16, filter_size=3, stride=1, pad='same')
#layer = Pool2DLayer(layer, 2)
for _ in range(18):
    layer = residual_block(layer, 16, survival_p=survival_p)
layer = residual_block(layer, 32, stride=2, survival_p=survival_p)
for _ in range(18):
    layer = residual_block(layer, 32, survival_p=survival_p)
layer = residual_block(layer, 64, stride=2, survival_p=survival_p)
for _ in range(18):
    layer = residual_block(layer, 64, survival_p=survival_p)
layer = Pool2DLayer(layer, pool_size=8, stride=1, mode="average_inc_pad")
layer = DenseLayer(layer, num_units=10, nonlinearity=softmax)


# In[64]:

for layer in get_all_layers(layer):
    print layer, layer.output_shape


# In[14]:

cifar10_loader = imp.load_source("cifar10_loader", "../papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py")


# In[24]:

#curr_dir = os.getcwd()
#os.chdir("../papers/deep_residual_learning/")
data = cifar10_loader.load_data()
X_train_and_valid, y_train_and_valid, X_test, y_test =     data["X_train"][0:50000], data["Y_train"][0:50000], data["X_test"], data["Y_test"]
#os.chdir(curr_dir)


# In[26]:

X_train = X_train_and_valid[ 0 : 0.9*X_train_and_valid.shape[0] ]
y_train = y_train_and_valid[ 0 : 0.9*y_train_and_valid.shape[0] ]
X_valid = X_train_and_valid[ 0.9*X_train_and_valid.shape[0] :: ]
y_valid = y_train_and_valid[ 0.9*y_train_and_valid.shape[0] :: ]


# In[27]:

X_train.shape


# In[51]:

X = T.tensor4('X')
y = T.ivector('y')

net_out = get_output(l_out, X)
net_out_det = get_output(l_out, X, deterministic=True)
loss = categorical_crossentropy(net_out, y).mean()
params = get_all_params(l_out, trainable=True)
grads = T.grad(loss, params)
updates = nesterov_momentum(grads, params, learning_rate=0.01, momentum=0.9)
train_fn = theano.function(inputs=[X, y], outputs=loss, updates=updates)
eval_fn = theano.function(inputs=[X, y], outputs=loss)
preds_fn = theano.function(inputs=[X], outputs=net_out_det)


# In[57]:

#X_train = X_train[0:10]
#y_train = y_train[0:10]
#X_valid = X_valid[0:10]
#y_valid = y_valid[0:10]


# In[56]:

batch_size = 128
n_batches = X_train.shape[0] // batch_size
num_epochs = 10
print "epoch,avg_train_loss,valid_loss,valid_acc,time"
for epoch in range(0, num_epochs):
    t0 = time()
    train_losses = []
    for b in range(0, n_batches):
        train_losses.append( train_fn(X_train[b*bs:(b+1)*bs], y_train[b*bs:(b+1)*bs]) )
    valid_loss = eval_fn(X_valid, y_valid)
    valid_preds = np.argmax(preds_fn(X_valid),axis=1)
    valid_acc = np.sum(valid_preds == y_valid)*1.0 / len(y_valid)
    print "%i,%f,%f,%f,%f" % (epoch+1, np.mean(train_losses), valid_loss, valid_acc, time()-t0)


# In[ ]:




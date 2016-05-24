import theano
from theano import tensor as T
from theano.ifelse import ifelse
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.regularization import *
from lasagne.random import get_rng
from lasagne.updates import *
from lasagne.init import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import imp
import os
from time import time

"""
   Binomial dropout layer

   Samples binomial(p,n=1) R.V. and multiplies the input tensor by
   this value. On its own, this layer is useless as it
   essentially either multiplies everything by one (i.e. do nothing),
   or it makes every value in the tensor zero (lose all information).
   This layer is intended to be used.
   
   Parameters
   ----------
   
   incoming : a :class:`Layer` instance
   p : float
       The survival probability for the input tensor

"""
class BinomialDropLayer(Layer):
    def __init__(self, incoming, p=0.5, **kwargs):
        super(BinomialDropLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return self.p*input
        else:
            #mask = self._srng.binomial(n=1, p=(self.p), size=(input.shape[0],),
            #    dtype=input.dtype)
            # apply the same thing to all examples in the minibatch
            mask = T.zeros((input.shape[0],)) + self._srng.binomial((1,), p=self.p, dtype=input.dtype)[0]
            mask = mask.dimshuffle(0,'x','x','x')
            return mask*input

class IfElseDropLayer(Layer):
    def __init__(self, incoming, p=0.5, **kwargs):
        super(IfElseDropLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return self.p*input
        else:
            return ifelse(
                T.lt(self._srng.uniform( (1,), 0, 1)[0], self.p),
                input,
                T.zeros(input.shape)
            )


"""
There is a difference between this residual block method and the one that is defined in:

https://github.com/Lasagne/Lasagne/issues/531

When the number of filters is different to the layer's output shape (or the stride is different),
instead of using a convolution to make things compatible, we use an average pooling with a pool
size of 1 and a the defined stride, followed by (if necessary) adding extra zero-padded feature
maps. This is because this is how the authors in the following link have defined it:

https://github.com/yueatsprograms/Stochastic_Depth/blob/master/ResidualDrop.lua
"""

def residual_block(layer, n_out_channels, stride=1, survival_p=0.5):
    conv = layer
    if stride > 1:
        layer = Pool2DLayer(layer, pool_size=1, stride=stride, mode="average_inc_pad")
    if (n_out_channels != layer.output_shape[1]):
        diff = n_out_channels-layer.output_shape[1]
        if diff % 2 == 0: 
            width_tp = ((diff/2, diff/2),)
        else:
            width_tp = (((diff/2)+1, diff/2),)
        layer = pad(layer, batch_ndim=1, width=width_tp)
    conv = Conv2DLayer(conv, num_filters=n_out_channels,
                       filter_size=(3,3), stride=(stride,stride), pad=(1,1), nonlinearity=linear)
    conv = BatchNormLayer(conv)
    conv = NonlinearityLayer(conv, nonlinearity=rectify)
    conv = Conv2DLayer(conv, num_filters=n_out_channels,
                       filter_size=(3,3), stride=(1,1), pad=(1,1), nonlinearity=linear)
    conv = BatchNormLayer(conv)
    conv = BinomialDropLayer(conv, p=survival_p)
    return NonlinearityLayer(ElemwiseSumLayer([conv, layer]), nonlinearity=rectify)

def get_net():
    # Architecture from:
    # https://github.com/yueatsprograms/Stochastic_Depth/blob/master/main.lua
    N = 18
    survival_p = 0.5
    layer = InputLayer( (None, 3, 32, 32) )
    layer = Conv2DLayer(layer, num_filters=16, filter_size=3, stride=1, pad='same')
    #layer = Pool2DLayer(layer, 2)
    for _ in range(N):
        layer = residual_block(layer, 16, survival_p=survival_p)
    layer = residual_block(layer, 32, stride=2, survival_p=survival_p)
    for _ in range(N):
        layer = residual_block(layer, 32, survival_p=survival_p)
    layer = residual_block(layer, 64, stride=2, survival_p=survival_p)
    for _ in range(N):
        layer = residual_block(layer, 64, survival_p=survival_p)
    layer = Pool2DLayer(layer, pool_size=8, stride=1, mode="average_inc_pad")
    layer = DenseLayer(layer, num_units=10, nonlinearity=softmax)
    for layer in get_all_layers(layer):
        print layer, layer.output_shape
    print "number of params:", count_params(layer)
    return layer

cifar10_loader = imp.load_source(
    "cifar10_loader", "../papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py")
curr_dir = os.getcwd()
os.chdir("../papers/deep_residual_learning/")
data = cifar10_loader.load_data()
X_train_and_valid, y_train_and_valid, X_test, y_test = \
    data["X_train"][0:50000], data["Y_train"][0:50000], data["X_test"], data["Y_test"]
os.chdir(curr_dir)

X_train = X_train_and_valid[ 0 : 0.9*X_train_and_valid.shape[0] ]
y_train = y_train_and_valid[ 0 : 0.9*y_train_and_valid.shape[0] ]
X_valid = X_train_and_valid[ 0.9*X_train_and_valid.shape[0] :: ]
y_valid = y_train_and_valid[ 0.9*y_train_and_valid.shape[0] :: ]

X = T.tensor4('X')
y = T.ivector('y')

layer = get_net()
net_out = get_output(layer, X)
net_out_det = get_output(layer, X, deterministic=True)
loss = categorical_crossentropy(net_out, y).mean()
params = get_all_params(layer, trainable=True)
grads = T.grad(loss, params)
updates = nesterov_momentum(grads, params, learning_rate=0.01, momentum=0.9)
train_fn = theano.function(inputs=[X, y], outputs=loss, updates=updates)
eval_fn = theano.function(inputs=[X, y], outputs=loss)
preds_fn = theano.function(inputs=[X], outputs=net_out_det)

bs = 128
n_batches = X_train.shape[0] // bs
num_epochs = 10
print "epoch,avg_train_loss,valid_loss,valid_acc,time"
for epoch in range(0, num_epochs):
    # shuffle examples
    idxs = [x for x in range(0, X_train.shape[0])]
    np.random.shuffle(idxs)
    X_train = X_train[idxs]
    y_train = y_train[idxs]
    train_losses = []
    t0 = time()
    for b in range(0, n_batches):
        train_losses.append( train_fn(X_train[b*bs:(b+1)*bs], y_train[b*bs:(b+1)*bs]) )
    valid_loss = eval_fn(X_valid, y_valid)
    valid_preds = np.argmax(preds_fn(X_valid),axis=1)
    valid_acc = np.sum(valid_preds == y_valid)*1.0 / len(y_valid)
    print "%i,%f,%f,%f,%f" % (epoch+1, np.mean(train_losses), valid_loss, valid_acc, time()-t0)

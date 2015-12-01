import numpy as np
from lasagne.layers import dnn
import lasagne as nn
import lasagne
import utils
import theano
import theano.tensor as T
from lasagne.layers.normalization import LocalResponseNormalization2DLayer
from lasagne.layers.base import Layer
import os
import pickle
import time


if __name__ == "__main__":
    import caffe

    net = caffe.Classifier('data/googlenet.prototxt', 'data/bvlc_googlenet.caffemodel')
    layer_names = net.blobs.keys()

    def get_caffe_params(l_name):
        layer_params = np.array(net.params[l_name])
        filter = caffe.io.blobproto_to_array(layer_params[0])
        bias = caffe.io.blobproto_to_array(layer_params[1])
        return utils.struct(
            filter=filter,
            bias=bias
        )

    def get_pretrained_params():
        l = []

        def append_name(name):
            l.append(get_caffe_params(name).filter)
            l.append(get_caffe_params(name).bias.reshape((-1,)))

        append_name('conv1/7x7_s2')
        append_name('conv2/3x3_reduce')
        append_name('conv2/3x3')

        for inception_layer in ['inception_3a', 'inception_3b',
                                'inception_4a', 'inception_4b', 'inception_4c', 'inception_4d', 'inception_4e',
                                'inception_5a', 'inception_5b'
                                ]:
            append_name(inception_layer + '/1x1')
            append_name(inception_layer + '/3x3_reduce')
            append_name(inception_layer + '/3x3')
            append_name(inception_layer + '/5x5_reduce')
            append_name(inception_layer + '/5x5')
            append_name(inception_layer + '/pool_proj')

        append_name('loss3/classifier')

        return l

    class flip(Layer):
        def get_output_shape_for(self, input_shape):
            return input_shape

        def get_output_for(self, input, **kwargs):
            return input[:, :, ::-1, ::-1]

else:
    # The following is only needed to exactly match caffe. Not needed in real-life scenario, override it.
    def flip(x):
        return x


def googlenet(batch_shape):
    """
    Create a googlenet, with the parameters from https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
    :param batch_shape: The shape of the input images. This should be of size (N, 3, 224, 224)
    :return: a struct with the input layer, the logit layer (before the final softmax) and the output layer.
    """
    l_in = lasagne.layers.InputLayer(
        shape=batch_shape,
        name='input',
    )

    l_conv1 = dnn.Conv2DDNNLayer(
        l_in,
        num_filters=64,
        filter_size=(7, 7),
        pad=3,
        stride=(2, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        name='conv1/7x7_s2',
    )

    l_pool1 = flip(dnn.MaxPool2DDNNLayer(
        flip(l_conv1),
        pool_size=(3, 3),  # pool_size
        stride=(2, 2),
        pad=(1, 1),
        name='pool1/3x3_s2'
    ))

    lrn = LocalResponseNormalization2DLayer(
        l_pool1,
        alpha=0.0001 / 5,
        beta=0.75,
        k=1,
        n=5,
        name='pool1/norm1',
    )

    l_conv2 = dnn.Conv2DDNNLayer(
        lrn,
        num_filters=64,
        filter_size=(1, 1),
        pad=0,
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        name='conv2/3x3_reduce',
    )

    l_conv2b = dnn.Conv2DDNNLayer(
        l_conv2,
        num_filters=192,
        filter_size=(3, 3),
        pad=1,
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.rectify,
        name='conv2/3x3',
    )

    lrn2 = LocalResponseNormalization2DLayer(
        l_conv2b,
        alpha=0.0001 / 5,
        beta=0.75,
        k=1,
        n=5,
        name='conv2/norm2',
    )

    l_pool2 = flip(dnn.MaxPool2DDNNLayer(
        flip(lrn2),
        pool_size=(3, 3),  # pool_size
        stride=(2, 2),
        pad=(1, 1),
        name='pool2/3x3_s2'
    ))

    def inception(layer, name, no_1x1=64, no_3x3r=96, no_3x3=128, no_5x5r=16, no_5x5=32, no_pool=32):
        l_conv_inc = dnn.Conv2DDNNLayer(
            layer,
            num_filters=no_1x1,
            filter_size=(1, 1),
            pad=0,
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name=name + '/1x1',
        )
        l_conv_inc2 = dnn.Conv2DDNNLayer(
            layer,
            num_filters=no_3x3r,
            filter_size=(1, 1),
            pad=0,
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name=name + '/3x3_reduce',
        )
        l_conv_inc2b = dnn.Conv2DDNNLayer(
            l_conv_inc2,
            num_filters=no_3x3,
            filter_size=(3, 3),
            pad=1,
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name=name + '/3x3',
        )
        l_conv_inc2c = dnn.Conv2DDNNLayer(
            layer,
            num_filters=no_5x5r,
            filter_size=(1, 1),
            pad=0,
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name=name + '/5x5_reduce',
        )
        l_conv_inc2d = dnn.Conv2DDNNLayer(
            l_conv_inc2c,
            num_filters=no_5x5,
            filter_size=(5, 5),
            pad=2,
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name=name + '/5x5',
        )
        l_pool2 = flip(dnn.MaxPool2DDNNLayer(
            flip(layer),
            pool_size=(3, 3),  # pool_size
            stride=(1, 1),
            pad=(1, 1),
            name=name + '/pool'
        ))
        l_conv_inc2e = dnn.Conv2DDNNLayer(
            l_pool2,
            num_filters=no_pool,
            filter_size=(1, 1),
            pad=0,
            stride=(1, 1),
            nonlinearity=lasagne.nonlinearities.rectify,
            name=name + '/pool_proj',
        )

        l_inc_out = nn.layers.concat([l_conv_inc, l_conv_inc2b, l_conv_inc2d, l_conv_inc2e])
        return l_inc_out

    l_inc_3a = inception(l_pool2, 'inception_3a', no_1x1=64, no_3x3r=96, no_3x3=128, no_5x5r=16, no_5x5=32, no_pool=32)
    l_inc_3b = inception(l_inc_3a, 'inception_3b', no_1x1=128, no_3x3r=128, no_3x3=192, no_5x5r=32, no_5x5=96,
                         no_pool=64)

    l_pool3 = flip(dnn.MaxPool2DDNNLayer(
        flip(l_inc_3b),
        pool_size=(3, 3),  # pool_size
        stride=(2, 2),
        pad=1,
        name='pool3/3x3_s2'
    ))

    l_inc_4a = inception(l_pool3, 'inception_4a', no_1x1=192, no_3x3r=96, no_3x3=208, no_5x5r=16, no_5x5=48, no_pool=64)
    l_inc_4b = inception(l_inc_4a, 'inception_4b', no_1x1=160, no_3x3r=112, no_3x3=224, no_5x5r=24, no_5x5=64,
                         no_pool=64)
    l_inc_4c = inception(l_inc_4b, 'inception_4c', no_1x1=128, no_3x3r=128, no_3x3=256, no_5x5r=24, no_5x5=64,
                         no_pool=64)
    l_inc_4d = inception(l_inc_4c, 'inception_4d', no_1x1=112, no_3x3r=144, no_3x3=288, no_5x5r=32, no_5x5=64,
                         no_pool=64)
    l_inc_4e = inception(l_inc_4d, 'inception_4e', no_1x1=256, no_3x3r=160, no_3x3=320, no_5x5r=32, no_5x5=128,
                         no_pool=128)

    l_pool4 = flip(dnn.MaxPool2DDNNLayer(
        flip(l_inc_4e),
        pool_size=(3, 3),  # pool_size
        stride=(2, 2),
        pad=1,
        name='pool4/3x3_s2'
    ))

    l_inc_5a = inception(l_pool4, 'inception_5a', no_1x1=256, no_3x3r=160, no_3x3=320, no_5x5r=32, no_5x5=128,
                         no_pool=128)
    l_inc_5b = inception(l_inc_5a, 'inception_5b', no_1x1=384, no_3x3r=192, no_3x3=384, no_5x5r=48, no_5x5=128,
                         no_pool=128)

    l_pool5 = flip(dnn.Pool2DDNNLayer(
        flip(l_inc_5b),
        pool_size=(7, 7),  # pool_size
        stride=(1, 1),
        pad=0,
        mode='average',
        name='pool5/7x7_s1'
    ))

    l_logit = dnn.Conv2DDNNLayer(
        l_pool5,
        num_filters=1000,
        filter_size=(1, 1),
        pad=0,
        stride=(1, 1),
        nonlinearity=lasagne.nonlinearities.linear,
        name='prob',
    )

    l_logit_flat = lasagne.layers.FlattenLayer(l_logit)
    l_dense = lasagne.layers.NonlinearityLayer(l_logit_flat, nonlinearity=lasagne.nonlinearities.softmax)
    l_out = l_dense

    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'googlenet.pkl')

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            nn.layers.set_all_param_values(l_dense, pickle.load(f))

    return utils.struct(
        input=l_in,
        logit=l_logit,
        out=l_out,
        interesting=l_inc_4c,
    )


if __name__ == "__main__":

    uptolayer = "prob"

    # now do the same in lasagne
    model = get_googlenet()
    for layer in nn.layers.get_all_layers(model.out):
        print "    %s %s" % (layer.name, layer.output_shape,)

    nn.layers.set_all_param_values(model.out, get_pretrained_params())

    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'googlenet.pkl')
    pickle.dump(nn.layers.get_all_param_values(model.out), open(filename, 'w'))

    x = nn.utils.shared_empty(dim=len(model.input.get_output_shape()))
    givens = {
        # target_var: T.sqr(y),
        model.input.input_var: x
    }
    idx = T.lscalar('idx')

    compute_output = theano.function([idx], model.out.get_output(deterministic=True), givens=givens,
                                     on_unused_input='ignore')

    lasagne_time = []
    caffe_time = []
    for i in xrange(10):
        inp = np.random.random((10, 3, 224, 224)).astype('float32')

        t_lasagne = time.time()
        x.set_value(inp)
        actual_output = compute_output(0)
        lasagne_time.append(time.time() - t_lasagne)
        print "lasagne time:", lasagne_time[-1]

        t_caffe = time.time()
        net.blobs['data'].data[...] = inp
        net.forward(end=uptolayer)
        goal_output = net.blobs[uptolayer].data
        caffe_time.append(time.time() - t_caffe)
        print "caffe time:", caffe_time[-1]

        print "goal_output shape:", goal_output.shape
        print "actual_output shape:", actual_output.shape
        print np.max(goal_output), np.max(actual_output)
        #print (np.count_nonzero( goal_output == np.max(goal_output) ),
        #        np.count_nonzero( actual_output == np.max(actual_output) ))
        print np.min(goal_output), np.min(actual_output)
        print np.sum(goal_output), np.sum(actual_output)
        print "0 =", np.max(abs(actual_output - goal_output)), "atol"
        print "0 =", np.max(abs(1.0 - actual_output / goal_output)), "rtol"

        #print goal_output[0,10,:5,:5]
        #print np.array(actual_output[0,10,:5,:5])
        print "Is this correct?", np.allclose(goal_output, actual_output, atol=1e-05)

    print "caffe", np.mean(caffe_time)
    print "lasagne", np.mean(lasagne_time)




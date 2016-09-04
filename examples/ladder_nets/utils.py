import numpy as np
import gzip
import cPickle as pickle

import lasagne
import theano.misc.pkl_utils
import theano.tensor as T

def pickle_load(f, encoding):
    return pickle.load(f)


def load_data(shared_var=True):
    """Get data with labels, split into training, validation and test set."""
    with gzip.open('./mnist.pkl.gz', 'rb') as f:
        data = pickle_load(f, encoding='latin-1')
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

    if shared_var:
        return dict(
            X_train=theano.shared(lasagne.utils.floatX(X_train)),
            y_train=T.cast(theano.shared(y_train), 'int32'),
            X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
            y_valid=T.cast(theano.shared(y_valid), 'int32'),
            X_test=theano.shared(lasagne.utils.floatX(X_test)),
            y_test=T.cast(theano.shared(y_test), 'int32'),
            num_examples_train=X_train.shape[0],
            num_examples_valid=X_valid.shape[0],
            num_examples_test=X_test.shape[0],
            input_dm=X_train.shape[1],
            output_dim=10,
        )
    else:
        return dict(
            X_train=np.float32(X_train),
            y_train=np.int32(y_train),
            X_valid=np.float32(X_valid),
            y_valid=np.int32(y_valid),
            X_test=np.float32(X_test),
            y_test=np.int32(y_test),
            num_examples_train=X_train.shape[0],
            num_examples_valid=X_valid.shape[0],
            num_examples_test=X_test.shape[0],
            input_dm=X_train.shape[1],
            output_dim=10,
        )


def softmax(vec, axis=1):
    """
     The ND implementation of softmax nonlinearity applied over a specified
     axis, which is by default the second dimension.
    """
    xdev = vec - vec.max(axis, keepdims=True)
    rval = T.exp(xdev)/(T.exp(xdev).sum(axis, keepdims=True))
    return rval

unzip = lambda zipped: zip(*zipped)
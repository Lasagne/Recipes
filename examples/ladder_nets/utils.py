# %load utils.py
import theano
import theano.tensor as T
import lasagne

import gzip
import cPickle as pickle
import sys

def pickle_load(f, encoding):
    return pickle.load(f)

def load_data():
    """Get data with labels, split into training, validation and test set."""
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        data = pickle_load(f, encoding='latin-1')
    X_train, y_train = data[0]
    X_valid, y_valid = data[1]
    X_test, y_test = data[2]

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
        input_dim=X_train.shape[1],
        output_dim=10,
    )
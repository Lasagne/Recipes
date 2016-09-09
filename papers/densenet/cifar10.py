#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides access to the CIFAR-10 dataset, including simple data augmentation.

Author: Jan Schlüter
"""
import os
import sys

import numpy as np


def download_dataset(path, source='https://www.cs.toronto.edu/~kriz/'
                                  'cifar-10-python.tar.gz'):
    """
    Downloads and extracts the dataset, if needed.
    """
    files = ['data_batch_%d' % (i + 1) for i in range(5)] + ['test_batch']
    for fn in files:
        if not os.path.exists(os.path.join(path, 'cifar-10-batches-py', fn)):
            break  # at least one file is missing
    else:
        return  # dataset is already complete

    print("Downloading and extracting %s into %s..." % (source, path))
    if sys.version_info[0] == 2:
        from urllib import urlopen
    else:
        from urllib.request import urlopen
    import tarfile
    if not os.path.exists(path):
        os.makedirs(path)
    u = urlopen(source)
    with tarfile.open(fileobj=u, mode='r|gz') as f:
        f.extractall(path=path)
    u.close()


def load_dataset(path):
    download_dataset(path)

    # training data
    data = [np.load(os.path.join(path, 'cifar-10-batches-py',
                                 'data_batch_%d' % (i + 1))) for i in range(5)]
    X_train = np.vstack([d['data'] for d in data])
    y_train = np.hstack([np.asarray(d['labels'], np.int8) for d in data])

    # test data
    data = np.load(os.path.join(path, 'cifar-10-batches-py', 'test_batch'))
    X_test = data['data']
    y_test = np.asarray(data['labels'], np.int8)

    # reshape
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    # normalize
    try:
        mean_std = np.load(os.path.join(path, 'cifar-10-mean_std.npz'))
        mean = mean_std['mean']
        std = mean_std['std']
    except IOError:
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        std = X_train.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
        np.savez(os.path.join(path, 'cifar-10-mean_std.npz'),
                 mean=mean, std=std)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    Generates one epoch of batches of inputs and targets, optionally shuffled.
    """
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def augment_minibatches(minibatches, flip=0.5, trans=4):
    """
    Randomly augments images by horizontal flipping with a probability of
    `flip` and random translation of up to `trans` pixels in both directions.
    """
    for inputs, targets in minibatches:
        batchsize, c, h, w = inputs.shape
        if flip:
            coins = np.random.rand(batchsize) < flip
            inputs = [inp[:, :, ::-1] if coin else inp
                      for inp, coin in zip(inputs, coins)]
            if not trans:
                inputs = np.asarray(inputs)
        outputs = inputs
        if trans:
            outputs = np.empty((batchsize, c, h, w), inputs[0].dtype)
            shifts = np.random.randint(-trans, trans, (batchsize, 2))
            for outp, inp, (x, y) in zip(outputs, inputs, shifts):
                if x > 0:
                    outp[:, :x] = 0
                    outp = outp[:, x:]
                    inp = inp[:, :-x]
                elif x < 0:
                    outp[:, x:] = 0
                    outp = outp[:, :x]
                    inp = inp[:, -x:]
                if y > 0:
                    outp[:, :, :y] = 0
                    outp = outp[:, :, y:]
                    inp = inp[:, :, :-y]
                elif y < 0:
                    outp[:, :, y:] = 0
                    outp = outp[:, :, :y]
                    inp = inp[:, :, -y:]
                outp[:] = inp
        yield outputs, targets

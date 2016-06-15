from fuel.datasets import MNIST
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, OneHotEncoding

import lasagne

from ladder_nets import build_cost, build_model

import numpy as np

import theano
import theano.tensor as T
import theano.misc.pkl_utils

import argparse
import cPickle


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-lr', '--learning_rate', type=float, default=0.1)
arg_parser.add_argument('-dlr', '--decrease_lr', type=float, default=1.)
arg_parser.add_argument('-bs', '--batch_size', type=int, default=100)
arg_parser.add_argument('-ep', '--max_epochs', type=int, default=15)
arg_parser.add_argument('-ctype', '--combinator', type=str, default='milaUDEM')
arg_parser.add_argument('-l', '--lambdas', type=str, default='0.1,0.1,0.1')
arg_parser.add_argument('-hdrop', '--hid_dropout', type=float, default=0.3)
args = arg_parser.parse_args()

NUM_EPOCHS = args.max_epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate

mnist = MNIST(which_sets=('train',), # sources='features',
              subset=slice(0, 50000), load_in_memory=True)
mnist_val = MNIST(which_sets=('train',), # sources='features',
                  subset=slice(50000, 60000), load_in_memory=True)
mnist_test = MNIST(which_sets=('test',), # sources='features',
                   load_in_memory=True)

data_stream = DataStream(mnist,
                         iteration_scheme=ShuffledScheme(mnist.num_examples,
                                                         batch_size=BATCH_SIZE))
data_stream_val = DataStream(mnist_val,
                             iteration_scheme=ShuffledScheme(
                                 mnist_val.num_examples, batch_size=BATCH_SIZE))
data_stream_test = DataStream(mnist_test,
                              iteration_scheme=ShuffledScheme(
                                  mnist_test.num_examples, batch_size=BATCH_SIZE))

data_stream = Flatten(data_stream, which_sources=('features',))
data_stream_val = Flatten(data_stream_val, which_sources=('features',))
data_stream_test = Flatten(data_stream_test, which_sources=('features',))

num_classes = 10

data_stream = OneHotEncoding(data_stream=data_stream,
                             which_sources=('targets',),
                             num_classes=num_classes)

data_stream_val = OneHotEncoding(data_stream=data_stream_val,
                                 which_sources=('targets',),
                                 num_classes=num_classes)

data_stream_test = OneHotEncoding(data_stream=data_stream_test,
                                  which_sources=('targets',),
                                  num_classes=num_classes)

# build network
num_encoder = [500, 10]
num_decoder = [500, 784]

[train_output_l, eval_output_l], dirty_net, clean_net = build_model(
    num_encoder, num_decoder, args.hid_dropout, args.hid_dropout, 
    batch_size=None, inp_size=784, combinator_type=args.combinator)

# print map(lambda x: [x.name, x.output_shape], dirty_net.values())
# print map(lambda x: [x.name, x.output_shape], clean_net.values())

# set up input/output variables
X = T.fmatrix('X')
y = T.imatrix('y')

# training output
output_train = lasagne.layers.get_output(train_output_l, X, deterministic=False)

# evaluation output. Also includes output of transform for plotting
output_eval = lasagne.layers.get_output(eval_output_l, X, deterministic=True)

# set up (possibly amortizable) lr, cost and updates
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

cost, rec_costs = build_cost(X, y, num_decoder, dirty_net, clean_net,
                             output_train, [float(x) for x in args.lambdas.split(',')])

net_params = lasagne.layers.get_all_params(train_output_l, trainable=True)
updates = lasagne.updates.adam(cost, net_params, learning_rate=sh_lr)

# get training and evaluation functions
train = theano.function([X, y], [cost + T.sum(rec_costs)] + rec_costs,
                        updates=updates)
eval = theano.function([X], [output_eval])

bl_name = 'enc_batchn_{}_learn'
means = [dirty_net[bl_name.format(i)].mean.ravel().mean() for i
         in range(len(num_encoder))]
means = T.stack(means, axis=1)
stds = [dirty_net[bl_name.format(i)].inv_std.ravel().mean() for i
        in range(len(num_encoder))]
stds = T.stack(stds, axis=1)
get_stats = theano.function([], [means, stds])
                            # , on_unused_input='ignore')

network_dump = {'train_output_layer': train_output_l,
                'eval_output_layer': eval_output_l,
                'dirty_net': dirty_net,
                'clean_net': clean_net,
                'x': X,
                'y': y,
                'output_eval': output_eval
                }


def save_dump(filename,param_values):
    f = file(filename, 'wb')
    cPickle.dump(param_values,f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def train_epoch(stream):
    costs = []
    rec_costs = []
    stats = []
    for batch in stream.get_epoch_iterator():
        train_out = train(*batch)
        stats.append(np.vstack(get_stats()))
        cur_cost = train_out[0]
        rec_cost = train_out[1:]

        costs.append(cur_cost)
        rec_costs.append(rec_cost)

    print '\n'.join(['Layer #{} rec cost: {}'.format(i, c) for i, c
                     in enumerate(np.mean(rec_costs, axis=0))])
    stats = np.stack(stats, axis=0).mean(axis=0)
    means, inv_stds = stat
    for i in range(len(num_encoder)):
        print '{}: mean {}, inv_std {}'.format(bl_name.format(i),
                                       np.allclose(means[i], 0.),
                                       np.allclose(inv_stds[i], 1.))
    return np.mean(costs)


def eval_epoch(stream, acc_only=True):
    preds = []
    targets = []
    for batch in stream.get_epoch_iterator():
        preds.extend(eval(batch[0]))
        targets.extend(batch[1])

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    acc = np.mean(preds.argmax(1) == targets.argmax(1)) # accuracy
    if not acc_only:
        nloglik = (np.log(preds) * targets).sum(1).mean()
        # confm = conf_mat(preds, targets)[0].astype(int)
        # CONF_MATS['iter_{}'.format(n)] = confm
        # save_dump('conf_mats_{}.pkl'.format(experiment_name), CONF_MATS)
        # print confm
        # return acc, nloglik, confm
    else:
        return acc
    
train_costs, train_accs, valid_accs = [], [], []
print 'Start training...'
try:
    for n in range(NUM_EPOCHS):
        train_costs.append(train_epoch(data_stream))
        train_accs.append(eval_epoch(data_stream))
        valid_accs.append(eval_epoch(data_stream_val))
        if (n+1) % 10 == 0:
            new_lr = sh_lr.get_value() * args.decrease_lr
            print "New LR:", new_lr
            sh_lr.set_value(lasagne.utils.floatX(new_lr))
        save_dump('accs_{}_ladder_net_mnist.pkl'.format(n),
                  zip(train_accs, valid_accs))
        # theano.misc.pkl_utils.dump(network_dump,
        #                            'iter_{}_ladder_nets_mnist.zip'.format(n))
        print "Epoch {}: Train cost {}, train acc {}, val acc {}".format(
                n, train_costs[-1], train_accs[-1], valid_accs[-1])
        # print 'TIMES: \ttrain {:10.2f}s, \tval {:10.2f}s'.format(t1-t0,
        #                                                          t2-t1)

    # TODO: needs an early stopping
except KeyboardInterrupt:
    pass

# save_dump('final_iter_{}_{}'.format(n, experiment_name),
#           lasagne.layers.get_all_param_values(output_layer))

theano.misc.pkl_utils.dump(network_dump,
                           'final_iter_{}_ladder_net_mnist.pkl'.format(n))
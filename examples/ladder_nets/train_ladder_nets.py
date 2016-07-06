from utils import load_data
from ladder_nets import *
import time
import theano.misc.pkl_utils
import lasagne
import cPickle
import numpy as np

LEARNING_RATE = 0.1
LR_DECREASE = 1.
BATCH_SIZE = 100
INPUT_SHAPE = [1, 28, 28]
NUM_EPOCHS = 15
COMBINATOR_TYPE = 'milaUDEM'
LAMBDAS = [1, 1, 1]
DROPOUT = 0.3
EXTRA_COST = False # True
ALPHAS = None, # [0.1]*3
BETAS = None, # [0.1]*3
NUM_LABELED = None
PSEUDO_LABELS = None
CONV = True # False
POOL = True # False

print "Loading data..."
dataset = load_data()

# build model
if CONV:
    input_shape = INPUT_SHAPE
    if POOL:
        num_encoder = [[40, 8, 0, 1, 2, 2], [10, 8, 0, 1, 2, 2]]
        num_decoder = [[40, 8, 0, 1, 2, 2], [1, 8, 0, 1, 2, 2]]
    else:
        num_encoder = [[40, 15, 0, 1], [10, 14, 0, 1]]
        num_decoder = [[40, 14, 0, 1], [1, 15, 0, 1]]
else:
    input_shape = np.prod(INPUT_SHAPE)
    num_encoder = [500, 10]
    num_decoder = [500, input_shape]

print "Building model and compiling functions..."
[train_output_l, eval_output_l], dirty_net, clean_net = build_model(
    num_encoder, num_decoder, DROPOUT, DROPOUT, input_shape=input_shape,
    combinator_type=COMBINATOR_TYPE, convolution=CONV, pooling=POOL)

print map(lambda x: (x.name, x.output_shape), dirty_net.values())

# set up input/output variables
X = T.fmatrix('X') if not CONV else T.ftensor4('x')
y = T.ivector('y')

# training output
output_train = lasagne.layers.get_output(train_output_l, X,
                                         deterministic=False).flatten(2)

# evaluation output. Also includes output of transform for plotting
output_eval = lasagne.layers.get_output(eval_output_l, X,
                                        deterministic=True).flatten(2)

# set up (possibly amortizable) lr, cost and updates
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

class_cost, rec_costs = build_cost(X, lasagne.utils.one_hot(y), num_decoder,
                                   dirty_net, clean_net, output_train,
                                   LAMBDAS, use_extra_costs=EXTRA_COST,
                                   alphas=ALPHAS, betas=BETAS,
                                   num_labeled=NUM_LABELED,
                                   pseudo_labels=PSEUDO_LABELS)
cost = class_cost + T.sum(rec_costs)
net_params = lasagne.layers.get_all_params(train_output_l, trainable=True)
updates = lasagne.updates.adam(cost, net_params, learning_rate=sh_lr)

# get training and evaluation functions, cost = class_cost + T.sum(rec_costs)
batch_index = T.iscalar('batch_index')
batch_slice = slice(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE)

pred = T.argmax(output_eval, axis=1)
accuracy = T.mean(T.eq(pred, y[:NUM_LABELED]), dtype=theano.config.floatX)

train = theano.function([batch_index], [cost] + rec_costs,
                        updates=updates, givens={
                            X: dataset['X_train'][batch_slice].reshape(
                               (-1, 1, 28, 28)
                           ),
                            y: dataset['y_train'][batch_slice],
                        })

eval = theano.function([batch_index], [cost, accuracy], givens={
                           X: dataset['X_valid'][batch_slice].reshape(
                               (-1, 1, 28, 28)
                           ),
                           y: dataset['y_valid'][batch_slice],
                       })

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


def train_epoch():
    costs = []
    rec_costs = []
    for b in range(num_batches_train):
        train_out = train(b)
        train_cost = train_out[0]
        rec_cost = train_out[1:]

        costs.append(train_cost)
        rec_costs.append(rec_cost)

    return (np.mean(costs), np.mean(rec_costs, axis=0))
    

def eval_epoch():
    costs = []
    accs = []
    for b in range(num_batches_valid):
        eval_cost, eval_acc = eval(b)
        costs.append(eval_cost)
        accs.append(eval_acc)

    return np.mean(eval_cost), np.mean(eval_acc)


num_batches_train = dataset['num_examples_train'] // BATCH_SIZE
num_batches_valid = dataset['num_examples_valid'] // BATCH_SIZE

train_costs, valid_costs, valid_accs = [], [], []

print "Starting training..."
now = time.time()

try:
    for n in range(NUM_EPOCHS):
        train_cost, rec_costs = train_epoch()
        eval_cost, acc = eval_epoch()
        
        train_costs.append(train_cost)
        valid_costs.append(eval_cost)
        valid_accs.append(acc)

        print "Epoch %d took %.3f s" % (n + 1, time.time() - now)
        now = time.time()
        print "Train cost {}, val cost {}, val acc {}".format(train_costs[-1], 
                                                              valid_costs[-1], 
                                                              valid_accs[-1])
        print '\n'.join(['Layer #{} rec cost: {}'.format(i, c) for i, c
                 in enumerate(rec_costs)])

        if (n+1) % 10 == 0:
            new_lr = sh_lr.get_value() * LR_DECREASE
            print "New LR:", new_lr
            sh_lr.set_value(lasagne.utils.floatX(new_lr))
except KeyboardInterrupt:
    pass

# uncomment if to save the learning curve
# save_dump('final_epoch_{}_accs_ladder_net_mnist.pkl'.format(n),
#                   zip(train_cost, valid_cost))

# uncomment if to save the params only
# save_dump('final_epoch_{}_ladder_net_mnist'.format(n),
#           lasagne.layers.get_all_param_values(output_layer))

# uncomment if to save the whole network
# theano.misc.pkl_utils.dump(network_dump,
#                            'final_epoch_{}_ladder_net_mnist.pkl'.format(n))
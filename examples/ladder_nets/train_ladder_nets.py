from __future__ import print_function
from utils import load_data
from ladder_nets import *
import time
import lasagne
import cPickle
import numpy as np

LEARNING_RATE = 0.1
LR_DECREASE = 1.
BATCH_SIZE = 100
INPUT_SHAPE = [1, 28, 28]
NUM_EPOCHS = 15
COMBINATOR_TYPE = 'milaUDEM' # or 'curiousAI'
DROPOUT = 0.3
EXTRA_COST = False # True
ALPHAS = None # [0.1]*3
BETAS = None # [0.1]*3
NUM_LABELED = None
PSEUDO_LABELS = None
CONV = True # False
POOL = True # False

print ("Loading data...")
dataset = load_data()

def get_encoder_settings(convolution, pooling):
    if convolution and pooling:
        settings = [('conv', (40, 8, 1, 0)), ('pool', (0, 2, 2, 0)),
                    ('conv', (10, 8, 1, 0)), ('pool', (0, 2, 2, 0))]
    elif convolution:
        settings = [('conv', (40, 15, 1, 0)), ('conv', (10, 14, 1, 0))]
    else:
        settings = [('dense', 500), ('dense', 10)]

    return settings

def get_decoder_settings(convolution, pooling):
    if convolution and pooling:
        settings = [('unpool', 'enc_3_pool'), ('deconv', (40, 8, 1, 0)),
                    ('unpool', 'enc_1_pool'), ('deconv', (1, 8, 1, 0))]
    elif convolution:
        settings = [('deconv', (40, 14, 1, 0)), ('deconv', (1, 15, 1, 0))]
    else:
        settings = [('dense', 10), ('dense', 784)]

    return settings

# build model
encoder_specs = get_encoder_settings(convolution=CONV, pooling=POOL)
decoder_specs = get_decoder_settings(convolution=CONV, pooling=POOL)
LAMBDAS = [1] * (len(decoder_specs) + 1)
input_shape = INPUT_SHAPE if CONV else np.prod(INPUT_SHAPE)

print ("Building model ...")
train_output_l, eval_output_l, dirty_encoder, dirty_decoder, clean_encoder = \
    build_model(encoder_specs, decoder_specs, DROPOUT, DROPOUT,
                input_shape=input_shape, combinator_type=COMBINATOR_TYPE)

print (map(lambda x: (x.name, x.output_shape), dirty_encoder.values()))
print (map(lambda x: (x.name, x.output_shape), dirty_decoder.values()))

# set up input/output variables
X = T.ftensor4('x') if CONV else T.fmatrix('X')
y = T.ivector('y')
y_onehot = lasagne.utils.one_hot(y, 10)

# training output
output_train = lasagne.layers.get_output(train_output_l, X,
                                         deterministic=False).flatten(2)

# evaluation output. Also includes output of transform for plotting
output_eval = lasagne.layers.get_output(eval_output_l, X,
                                        deterministic=True).flatten(2)

# set up (possibly amortizable) lr, cost and updates
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))

print ("Building costs and updates ...")
class_cost, stats = build_costNstats(y_onehot, output_train, output_eval,
                                     NUM_LABELED, PSEUDO_LABELS)

rec_costs = build_rec_costs(X, clean_encoder, dirty_decoder, decoder_specs,
                            lambdas=LAMBDAS, alphas=ALPHAS, betas=BETAS,
                            use_extra_costs=EXTRA_COST)

cost = class_cost + T.sum(rec_costs)
net_params = lasagne.layers.get_all_params(train_output_l, trainable=True)
updates = lasagne.updates.adam(cost, net_params, learning_rate=sh_lr)

# get training and evaluation functions, cost = class_cost + T.sum(rec_costs)
batch_index = T.iscalar('batch_index')
batch_slice = slice(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE)

print ("Compiling functions...")
train = theano.function([batch_index], [cost] + rec_costs,
                        updates=updates, givens={
                            X: dataset['X_train'][batch_slice].reshape(
                               (-1,) + tuple(input_shape)
                           ),
                            y: dataset['y_train'][batch_slice],
                        })

eval = theano.function([batch_index], [cost] + stats, givens={
                           X: dataset['X_valid'][batch_slice].reshape(
                               (-1,) + tuple(input_shape)
                           ),
                           y: dataset['y_valid'][batch_slice],
                       })

network_dump = {'train_output_layer': train_output_l,
                'eval_output_layer': eval_output_l,
                'dirty_net': dirty_decoder,
                'clean_net': clean_encoder,
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

print ("Starting training...")
now = time.time()

try:
    for n in range(NUM_EPOCHS):
        train_cost, rec_costs = train_epoch()
        eval_cost, acc = eval_epoch()
        
        train_costs.append(train_cost)
        valid_costs.append(eval_cost)
        valid_accs.append(acc)

        print ("Epoch %d took %.3f s" % (n + 1, time.time() - now))
        now = time.time()
        print ("Train cost {}, val cost {}, val acc {}".format(train_costs[-1],
                                                              valid_costs[-1],
                                                              valid_accs[-1]))
        print ('\n'.join(['Layer #{} rec cost: {}'.format(i, c) for i, c
                 in enumerate(rec_costs)]))

        if (n+1) % 10 == 0:
            new_lr = sh_lr.get_value() * LR_DECREASE
            print ("New LR:", new_lr)
            sh_lr.set_value(lasagne.utils.floatX(new_lr))
except KeyboardInterrupt:
    pass

# uncomment if to save the learning curve
# save_dump('final_epoch_{}_accs_ladder_net_mnist.pkl'.format(n),
#                   zip(train_cost, valid_cost))

# uncomment if to save the params only
# save_dump('final_epoch_{}_ladder_net_mnist.pkl'.format(n),
#           lasagne.layers.get_all_param_values(output_layer))

# uncomment if to save the whole network
# save_dump('final_epoch_{}_ladder_net_mnist.pkl'.format(n),
#           network_dump)
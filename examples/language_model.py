"""
Examples that mimick the setup in http://arxiv.org/abs/1409.2329
except that we use GRU instead of a LSTM layers.

The example demonstrates:

    * How to setup GRU in Lasagne to predict a target for every position in a
      sequence.
    * How to setup Lasagne for language modelling tasks.
    * Fancy reordering of data to allow theano to process a large text
      corpus.
    * How to combine recurrent and feed-forward layers in the same Lasagne
      model.
"""
from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import os
import time
import gzip
import lasagne

#  SETTINGS
folder = 'penntree'                 # subfolder with data
BATCH_SIZE = 20                     # batch size
MODEL_SEQ_LEN = 20                  # how many steps to unroll
TOL = 1e-6                          # numerial stability
INI = lasagne.init.Uniform(0.1)     # initial parameter values
REC_NUM_UNITS = 200                 # number of LSTM units
embedding_size = 200                # Embedding size
dropout_frac = 0                    # optional recurrent dropout
lr = 1.0                            # learning rate
decay = 2.0                         # decay factor
no_decay_epochs = 5                 # run this many epochs before first decay
max_grad_norm = 10                  # scale steps if norm is above this value
num_epochs = 200                    # Number of epochs to run


# First we'll define a functions to load the Penn Tree data.

def load_data(file_name, vocab_map, vocab_idx):
    """
    Loads Penn Tree files downloaded from https://github.com/wojzaremba/lstm

    Parameters
    ----------
    file_name : str
        Path to Penn tree file.
    vocab_map : dictionary
        Dictionary mapping words to integers
    vocab_idx : one element list
        Current vocabulary index.

    Returns
    -------
    Returns an array with each words specified in file_name as integers.
    Note that the function has the side effects that vocab_map and vocab_idx
    are updated.

    Notes
    -----
    This is python port of the LUA function load_data in
    https://github.com/wojzaremba/lstm/blob/master/data.lua
    """
    def process_line(line):
        line = line.lstrip()
        line = line.replace('\n', '<eos>')
        words = line.split(" ")
        if words[-1] == "":
            del words[-1]
        return words

    words = []
    with gzip.open(file_name, 'rb') as f:
        for line in f.readlines():
            words += process_line(line)

    n_words = len(words)
    print("Loaded %i words from %s" % (n_words, file_name))

    x = np.zeros(n_words)
    for wrd_idx, wrd in enumerate(words):
        if wrd not in vocab_map:
            vocab_map[wrd] = vocab_idx[0]
            vocab_idx[0] += 1
        x[wrd_idx] = vocab_map[wrd]
    return x.astype('int32')


def reorder(x_in, batch_size, model_seq_len):
    """
    Rearranges data set so batches process sequential data.
    If we have the dataset:

    x_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

    and the batch size is 2 and the model_seq_len is 3. Then the dataset is
    reordered such that:

                   Batch 1    Batch 2
                 ------------------------
    batch pos 1  [1, 2, 3]   [4, 5, 6]
    batch pos 2  [7, 8, 9]   [10, 11, 12]

    This ensures that we use the last hidden state of batch 1 to initialize
    batch 2.

    Also creates targets. In language modelling the target is to predict the
    next word in the sequence.

    Parameters
    ----------
    x_in : 1D numpy.array
    batch_size : int
    model_seq_len : int
        number of steps the model is unrolled

    Returns
    -------
    reordered x_in and reordered targets. Targets are shifted version of x_in.

    """
    if x_in.ndim != 1:
        raise ValueError("Data must be 1D, was", x_in.ndim)

    if x_in.shape[0] % (batch_size*model_seq_len) == 0:
        print(" x_in.shape[0] % (batch_size*model_seq_len) == 0 -> x_in is "
              "set to x_in = x_in[:-1]")
        x_in = x_in[:-1]

    x_resize =  \
        (x_in.shape[0] // (batch_size*model_seq_len))*model_seq_len*batch_size
    n_samples = x_resize // (model_seq_len)
    n_batches = n_samples // batch_size

    targets = x_in[1:x_resize+1].reshape(n_samples, model_seq_len)
    x_out = x_in[:x_resize].reshape(n_samples, model_seq_len)

    out = np.zeros(n_samples, dtype=int)
    for i in range(n_batches):
        val = range(i, n_batches*batch_size+i, n_batches)
        out[i*batch_size:(i+1)*batch_size] = val

    x_out = x_out[out]
    targets = targets[out]

    return x_out.astype('int32'), targets.astype('int32')


def traindata(model_seq_len, batch_size, vocab_map, vocab_idx):
    x = load_data(os.path.join(folder, "ptb.train.txt.gz"),
                  vocab_map, vocab_idx)
    return reorder(x, batch_size, model_seq_len)


def validdata(model_seq_len, batch_size, vocab_map, vocab_idx):
    x = load_data(os.path.join(folder, "ptb.valid.txt.gz"),
                  vocab_map, vocab_idx)
    return reorder(x, batch_size, model_seq_len)

# vocab_map and vocab_idx are updated as side effects of load_data
vocab_map = {}
vocab_idx = [0]
x_train, y_train = traindata(MODEL_SEQ_LEN, BATCH_SIZE, vocab_map, vocab_idx)
x_valid, y_valid = validdata(MODEL_SEQ_LEN, BATCH_SIZE, vocab_map, vocab_idx)
vocab_size = vocab_idx[0]


print("-" * 80)
print("Vocab size:s", vocab_size)
print("Data shapes")
print("Train data:", x_train.shape)
print("Valid data:", x_valid.shape)
print("-" * 80)

# Theano symbolic vars
sym_x = T.imatrix()
sym_y = T.imatrix()

# symbolic vars for initial recurrent initial states
hid1_init_sym = T.matrix()
hid2_init_sym = T.matrix()


# BUILDING THE MODEL
# Model structure:
#
#    embedding --> GRU1 --> GRU2 --> output network --> predictions
l_inp = lasagne.layers.InputLayer((BATCH_SIZE, MODEL_SEQ_LEN))

l_emb = lasagne.layers.EmbeddingLayer(
    l_inp,
    input_size=vocab_size,       # size of embedding = number of words
    output_size=embedding_size,  # vector size used to represent each word
    W=INI)

l_drp0 = lasagne.layers.DropoutLayer(l_emb, p=dropout_frac)


def create_gate():
    return lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=None)

# first GRU layer
l_rec1 = lasagne.layers.GRULayer(
    l_drp0,
    num_units=REC_NUM_UNITS,
    resetgate=create_gate(),
    updategate=create_gate(),
    hidden_update=create_gate(),
    learn_init=False,
    hid_init=hid1_init_sym)

l_drp1 = lasagne.layers.DropoutLayer(l_rec1, p=dropout_frac)

# Second GRU layer
l_rec2 = lasagne.layers.GRULayer(
    l_drp1,
    num_units=REC_NUM_UNITS,
    resetgate=create_gate(),
    updategate=create_gate(),
    hidden_update=create_gate(),
    learn_init=False,
    hid_init=hid2_init_sym)

l_drp2 = lasagne.layers.DropoutLayer(l_rec2, p=dropout_frac)


# by reshaping we can combine feed-forward and recurrent layers in the
# same Lasagne model.
l_shp = lasagne.layers.ReshapeLayer(l_drp2,
                                    (BATCH_SIZE*MODEL_SEQ_LEN, REC_NUM_UNITS))
l_out = lasagne.layers.DenseLayer(l_shp,
                                  num_units=vocab_size,
                                  nonlinearity=lasagne.nonlinearities.softmax)
l_out = lasagne.layers.ReshapeLayer(l_out,
                                    (BATCH_SIZE, MODEL_SEQ_LEN, vocab_size))


def calc_cross_ent(net_output, targets):
    # Helper function to calculate the cross entropy error
    preds = T.reshape(net_output, (BATCH_SIZE * MODEL_SEQ_LEN, vocab_size))
    preds += TOL  # add constant for numerical stability
    targets = T.flatten(targets)
    cost = T.nnet.categorical_crossentropy(preds, targets)
    return cost

# Note the use of deterministic keyword to disable dropout during evaluation.
train_out,  l_rec1_hid_out,  l_rec2_hid_out = lasagne.layers.get_output(
    [l_out, l_rec1, l_rec2], sym_x, deterministic=False)


# after we have called get_ouput then the layers will have reference to
# their output values. We need to keep track of the output values for both
# training and evaluation and for each hidden layer because we want to
# initialze each batch with the last hidden values from the previous batch.
hidden_states_train = [l_rec1_hid_out, l_rec2_hid_out]

eval_out, l_rec1_hid_out,  l_rec2_hid_out = lasagne.layers.get_output(
    [l_out, l_rec1, l_rec2], sym_x, deterministic=True)
hidden_states_eval = [l_rec1_hid_out, l_rec2_hid_out]

cost_train = T.mean(calc_cross_ent(train_out, sym_y))
cost_eval = T.mean(calc_cross_ent(eval_out, sym_y))

# Get list of all trainable parameters in the network.
all_params = lasagne.layers.get_all_params(l_out, trainable=True)

# Calculate gradients w.r.t cost function. Note that we scale the cost with
# MODEL_SEQ_LEN. This is to be consistent with
# https://github.com/wojzaremba/lstm . The scaling is due to difference
# between torch and theano. We could have also scaled the learning rate, and
# also rescaled the norm constraint.
all_grads = T.grad(cost_train*MODEL_SEQ_LEN, all_params)

# With the gradients for each parameter we can calculate update rules for each
# parameter. Lasagne implements a number of update rules, here we'll use
# sgd and a total_norm_constraint.
all_grads, norm = lasagne.updates.total_norm_constraint(
    all_grads, max_grad_norm, return_norm=True)

# Use shared variable for learning rate. Allows us to change the learning rate
# during training.
sh_lr = theano.shared(lasagne.utils.floatX(lr))
updates = lasagne.updates.sgd(all_grads, all_params, learning_rate=sh_lr)

# Define evaluation function. This graph disables dropout.
print("compiling f_eval...")
fun_inp = [sym_x, sym_y, hid1_init_sym, hid2_init_sym]
f_eval = theano.function(fun_inp,
                         [cost_eval,
                          hidden_states_eval[0][:, -1],
                          hidden_states_eval[1][:, -1]])

# define training function. This graph has dropout enabled.
# The update arg specifies that the parameters should be updated using the
# update rules.
print("compiling f_train...")
f_train = theano.function(fun_inp,
                          [cost_train,
                           norm,
                           hidden_states_train[0][:, -1],
                           hidden_states_train[1][:, -1]],
                          updates=updates)


def calc_perplexity(x, y):
    """
    Helper function to evaluate perplexity.

    Perplexity is the inverse probability of the test set, normalized by the
    number of words.
    See: https://web.stanford.edu/class/cs124/lec/languagemodeling.pdf

    This function is largely based on the perplexity calcualtion from
    https://github.com/wojzaremba/lstm/
    """

    n_batches = x.shape[0] // BATCH_SIZE
    l_cost = []
    hid1, hid2 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                           dtype='float32') for _ in range(2)]

    for i in range(n_batches):
        x_batch = x[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        y_batch = y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        cost, hid1, hid2 = f_eval(
            x_batch, y_batch, hid1, hid2)
        l_cost.append(cost)

    n_words_evaluated = (x.shape[0] - 1) / MODEL_SEQ_LEN
    perplexity = np.exp(np.sum(l_cost) / n_words_evaluated)

    return perplexity

n_batches_train = x_train.shape[0] // BATCH_SIZE
for epoch in range(num_epochs):
    l_cost, l_norm, batch_time = [], [], time.time()

    # use zero as initial state
    hid1, hid2 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS),
                           dtype='float32') for _ in range(2)]
    for i in range(n_batches_train):
        x_batch = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]   # single batch
        y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        cost, norm, hid1, hid2 = f_train(
            x_batch, y_batch, hid1, hid2)
        l_cost.append(cost)
        l_norm.append(norm)

    if epoch > (no_decay_epochs - 1):
        current_lr = sh_lr.get_value()
        sh_lr.set_value(lasagne.utils.floatX(current_lr / float(decay)))

    elapsed = time.time() - batch_time
    words_per_second = float(BATCH_SIZE*(MODEL_SEQ_LEN)*len(l_cost)) / elapsed
    n_words_evaluated = (x_train.shape[0] - 1) / MODEL_SEQ_LEN
    perplexity_valid = calc_perplexity(x_valid, y_valid)
    perplexity_train = np.exp(np.sum(l_cost) / n_words_evaluated)
    print("Epoch           :", epoch)
    print("Perplexity Train:", perplexity_train)
    print("Perplexity valid:", perplexity_valid)
    print("Words per second:", words_per_second)
    l_cost = []
    batch_time = 0
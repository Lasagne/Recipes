'''
Recurrent network example.  Trains a 2 layered LSTM network to learn
text from a user-provided input file. The network can then be used to generate
text using a short string as seed (refer to the variable generation_phrase).
This example is partly based on Andrej Karpathy's blog
(http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
and a similar example in the Keras package (keras.io).
The inputs to the network are batches of sequences of characters and the corresponding
targets are the characters in the text shifted to the right by one. 
Assuming a sequence length of 5, a training point for a text file
"The quick brown fox jumps over the lazy dog" would be
INPUT : 'T','h','e',' ','q'
OUTPUT: 'u'

The loss function compares (via categorical crossentropy) the prediction
with the output/target.

Also included is a function to generate text using the RNN given the first 
character.  

About 20 or so epochs are necessary to generate text that "makes sense".

Written by @keskarnitish
Pre-processing of text uses snippets of Karpathy's code (BSD License)
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
import urllib2 #For downloading the sample text file. You won't need this if you are providing your own file.
try:
    in_text = urllib2.urlopen('https://s3.amazonaws.com/text-datasets/nietzsche.txt').read()
    #You can also use your own file
    #The file must be a simple text file.
    #Simply edit the file name below and uncomment the line.  
    #in_text = open('your_file.txt', 'r').read()
    in_text = in_text.decode("utf-8-sig").encode("utf-8")
except Exception as e:
    print("Please verify the location of the input file/URL.")
    print("A sample txt file can be downloaded from https://s3.amazonaws.com/text-datasets/nietzsche.txt")
    raise IOError('Unable to Read Text')

generation_phrase = "The quick brown fox jumps" #This phrase will be used as seed to generate text.

#This snippet loads the text file and creates dictionaries to 
#encode characters into a vector-space representation and vice-versa. 
chars = list(set(in_text))
data_size, vocab_size = len(in_text), len(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 20

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 1000

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 128


def gen_data(p, batch_size = BATCH_SIZE, data=in_text, return_target=True):
    '''
    This function produces a semi-redundant batch of training samples from the location 'p' in the provided string (data).
    For instance, assuming SEQ_LENGTH = 5 and p=0, the function would create batches of 
    5 characters of the string (starting from the 0th character and stepping by 1 for each semi-redundant batch)
    as the input and the next character as the target.
    To make this clear, let us look at a concrete example. Assume that SEQ_LENGTH = 5, p = 0 and BATCH_SIZE = 2
    If the input string was "The quick brown fox jumps over the lazy dog.",
    For the first data point,
    x (the inputs to the neural network) would correspond to the encoding of 'T','h','e',' ','q'
    y (the targets of the neural network) would be the encoding of 'u'
    For the second point,
    x (the inputs to the neural network) would correspond to the encoding of 'h','e',' ','q', 'u'
    y (the targets of the neural network) would be the encoding of 'i'
    The data points are then stacked (into a three-dimensional tensor of size (batch_size,SEQ_LENGTH,vocab_size))
    and returned. 
    Notice that there is overlap of characters between the batches (hence the name, semi-redundant batch).
    '''
    x = np.zeros((batch_size,SEQ_LENGTH,vocab_size))
    y = np.zeros(batch_size)

    for n in range(batch_size):
        ptr = n
        for i in range(SEQ_LENGTH):
            x[n,i,char_to_ix[data[p+ptr+i]]] = 1.
        if(return_target):
            y[n] = char_to_ix[data[p+ptr+SEQ_LENGTH]]
    return x, np.array(y,dtype='int32')



def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
   
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(None, None, vocab_size))

    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients. 

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh)

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.tanh,
        only_return_final=True)

    # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the softmax nonlinearity to 
    # create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    l_out = lasagne.layers.DenseLayer(l_forward_2, num_units=vocab_size, W = lasagne.init.Normal(), nonlinearity=lasagne.nonlinearities.softmax)

    # Theano tensor for the targets
    target_values = T.ivector('target_output')
    
    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs. 
    
    probs = theano.function([l_in.input_var],network_output,allow_input_downcast=True)

    # The next function generates text given a phrase of length at least SEQ_LENGTH.
    # The phrase is set using the variable generation_phrase.
    # The optional input "N" is used to set the number of characters of text to predict. 

    def try_it_out(N=200):
        '''
        This function uses the user-provided string "generation_phrase" and current state of the RNN generate text.
        The function works in three steps:
        1. It converts the string set in "generation_phrase" (which must be over SEQ_LENGTH characters long) 
           to encoded format. We use the gen_data function for this. By providing the string and asking for a single batch,
           we are converting the first SEQ_LENGTH characters into encoded form. 
        2. We then use the LSTM to predict the next character and store it in a (dynamic) list sample_ix. This is done by using the 'probs'
           function which was compiled above. Simply put, given the output, we compute the probabilities of the target and pick the one 
           with the highest predicted probability. 
        3. Once this character has been predicted, we construct a new sequence using all but first characters of the 
           provided string and the predicted character. This sequence is then used to generate yet another character.
           This process continues for "N" characters. 
        To make this clear, let us again look at a concrete example. 
        Assume that SEQ_LENGTH = 5 and generation_phrase = "The quick brown fox jumps". 
        We initially encode the first 5 characters ('T','h','e',' ','q'). The next character is then predicted (as explained in step 2). 
        Assume that this character was 'J'. We then construct a new sequence using the last 4 (=SEQ_LENGTH-1) characters of the previous
        sequence ('h','e',' ','q') , and the predicted letter 'J'. This new sequence is then used to compute the next character and 
        the process continues.
        '''

        assert(len(generation_phrase)>=SEQ_LENGTH)
        sample_ix = []
        x,_ = gen_data(len(generation_phrase)-SEQ_LENGTH, 1, generation_phrase,0)

        for i in range(N):
            # Pick the character that got assigned the highest probability
            ix = np.argmax(probs(x).ravel())
            # Alternatively, to sample from the distribution instead:
            # ix = np.random.choice(np.arange(vocab_size), p=probs(x).ravel())
            sample_ix.append(ix)
            x[:,0:SEQ_LENGTH-1,:] = x[:,1:,:]
            x[:,SEQ_LENGTH-1,:] = 0
            x[0,SEQ_LENGTH-1,sample_ix[-1]] = 1. 

        random_snippet = generation_phrase + ''.join(ix_to_char[ix] for ix in sample_ix)    
        print("----\n %s \n----" % random_snippet)


    
    print("Training ...")
    print("Seed used for text generation is: " + generation_phrase)
    p = 0
    try:
        for it in xrange(data_size * num_epochs / BATCH_SIZE):
            try_it_out() # Generate text using the p^th character as the start. 
            
            avg_cost = 0;
            for _ in range(PRINT_FREQ):
                x,y = gen_data(p)
                
                #print(p)
                p += SEQ_LENGTH + BATCH_SIZE - 1 
                if(p+BATCH_SIZE+SEQ_LENGTH >= data_size):
                    print('Carriage Return')
                    p = 0;
                

                avg_cost += train(x, y)
            print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))
                    
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()

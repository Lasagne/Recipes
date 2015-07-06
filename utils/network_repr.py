import lasagne
from lasagne.layers import get_all_layers
from collections import deque, defaultdict


def get_network_str(layer, get_network=True, incomings=False, outgoings=False):
    """ Returns a string representation of the entire network contained under this layer.

        Parameters
        ----------
        layer : Layer or list
            the :class:`Layer` instance for which to gather all layers feeding
            into it, or a list of :class:`Layer` instances.

        get_network : boolean
            if True, calls `get_all_layers` on `layer`
            if False, assumes `layer` already contains all `Layer` instances intended for representation

        incomings : boolean
            if True, representation includes a list of all incomings for each `Layer` instance

        outgoings: boolean
            if True, representation includes a list of all outgoings for each `Layer` instance

        Returns
        -------
        str
            A string representation of `layer`. Each layer is assigned an ID which is it's corresponding index
            in the list obtained from `get_all_layers`.
        """

    # `layer` can either be a single `Layer` instance or a list of `Layer` instances.
    # If list, it can already be the result from `get_all_layers` or not, indicated by the `get_network` flag
    # Get network using get_all_layers if required:
    if get_network:
        network = get_all_layers(layer)
    else:
        network = layer

    # Initialize a list of lists to (temporarily) hold the str representation of each component, insert header
    network_str = deque([])
    network_str = _insert_header(network_str, incomings=incomings, outgoings=outgoings)

    # The representation can optionally display incoming and outgoing layers for each layer, similar to adjacency lists.
    # If requested (using the incomings and outgoings flags), build the adjacency lists.
    # The numbers/ids in the adjacency lists correspond to the layer's index in `network`
    if incomings or outgoings:
        ins, outs = _get_adjacency_lists(network)

    # For each layer in the network, build a representation and append to `network_str`
    for i, current_layer in enumerate(network):

        # Initialize list to (temporarily) hold str of layer
        layer_str = deque([])

        # First column for incomings, second for the layer itself, third for outgoings, fourth for layer description
        if incomings:
            layer_str.append(ins[i])
        layer_str.append(i)
        if outgoings:
            layer_str.append(outs[i])
        layer_str.append(str(current_layer))    # default representation can be changed by overriding __str__
        network_str.append(layer_str)
    return _get_table_str(network_str)


def _insert_header(network_str, incomings, outgoings):
    """ Insert the header (first two lines) in the representation."""
    line_1 = deque([])
    if incomings:
        line_1.append('In -->')
    line_1.append('Layer')
    if outgoings:
        line_1.append('--> Out')
    line_1.append('Description')
    line_2 = deque([])
    if incomings:
        line_2.append('-------')
    line_2.append('-----')
    if outgoings:
        line_2.append('-------')
    line_2.append('-----------')
    network_str.appendleft(line_2)
    network_str.appendleft(line_1)
    return network_str


def _get_adjacency_lists(network):
    """ Returns adjacency lists for each layer (node) in network.
        Warning: Assumes repr is unique to a layer instance, else this entire approach WILL fail."""
    # ins  is a dict, keys are layer indices and values are lists of incoming layer indices
    # outs is a dict, keys are layer indices and values are lists of outgoing layer indices
    ins = defaultdict(list)
    outs = defaultdict(list)
    lookup = {repr(layer): index for index, layer in enumerate(network)}

    for current_layer in network:
        if hasattr(current_layer, 'input_layers'):
            layer_ins = current_layer.input_layers
        elif hasattr(current_layer, 'input_layer'):
            layer_ins = [current_layer.input_layer]
        else:
            layer_ins = []

        ins[lookup[repr(current_layer)]].extend([lookup[repr(l)] for l in layer_ins])

        for l in layer_ins:
            outs[lookup[repr(l)]].append(lookup[repr(current_layer)])
    return ins, outs


def _get_table_str(table):
    """ Pretty print a table provided as a list of lists."""
    table_str = ''
    col_size = [max(len(str(val)) for val in column) for column in zip(*table)]
    for line in table:
        table_str += '\n'
        table_str += '    '.join('{0:<{1}}'.format(val, col_size[i]) for i, val in enumerate(line))
    return table_str


def example1():
    """ Sequential network, no branches or cycles"""
    l_in = lasagne.layers.InputLayer((100, 20))
    l_hidden1 = lasagne.layers.DenseLayer(l_in, num_units=20)
    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1)
    l_hidden2 = lasagne.layers.DenseLayer(l_hidden1_dropout, num_units=20)
    l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2)
    l_out = lasagne.layers.DenseLayer(l_hidden2_dropout, num_units=10)
    print(get_network_str(l_out))
    return None


def example2():
    """ Two branches"""
    # Input
    l_in = lasagne.layers.InputLayer((100, 1, 20, 20))
    # Branch one
    l_conv1 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5, 5))
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2, 2))
    l_dense1 = lasagne.layers.DenseLayer(l_pool1, num_units=20)
    # Branch two
    l_conv2 = lasagne.layers.Conv2DLayer(l_in, num_filters=32, filter_size=(5, 5))
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2, 2))
    l_dense2 = lasagne.layers.DenseLayer(l_pool2, num_units=20)
    # Merge
    l_concat = lasagne.layers.ConcatLayer((l_dense1, l_dense2))
    # Output
    l_out = lasagne.layers.DenseLayer(l_concat, num_units=10)
    layers = get_all_layers(l_out)
    print(get_network_str(layers, get_network=False, incomings=True, outgoings=True))
    return None


def main():
    print('===========================================================')
    example1()
    print('===========================================================')
    example2()
    print('===========================================================')
    return None

if __name__ == '__main__':
    main()
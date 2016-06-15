from lasagne.layers import InputLayer, MergeLayer, DenseLayer, DropoutLayer, \
    GaussianNoiseLayer, NonlinearityLayer
from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import softmax, linear, tanh, rectify
import lasagne
import theano
from theano import tensor as T
import numpy as np
from collections import OrderedDict


def _milaUDEM_params(shape, name):
    values = np.zeros((6,) + shape, dtype=theano.config.floatX)

    b_lin = theano.shared(values[0],
                               name='bias_lin_{}'.format(name))
    b_sigm = theano.shared(values[1],
                                name='bias_sigm_{}'.format(name))

    w_u_lin = theano.shared(values[2],
                                 name='weight_u_lin_{}'.format(name))
    w_u_sigm = theano.shared(values[3],
                                  name='weight_u_sigm_{}'.format(name))
    w_zu_lin = theano.shared(values[4],
                                  name='weight_zu_lin_{}'.format(name))
    w_zu_sigm = theano.shared(values[5],
                                   name='weight_zu_sigm_{}'.format(name))

    values = np.ones((3,) + shape, dtype=theano.config.floatX)
    w_z_lin = theano.shared(values[0],
                                 name='weight_z_lin_{}'.format(name))
    w_z_sigm = theano.shared(values[1],
                                  name='weight_z_sigm_{}'.format(name))
    w_sigm = theano.shared(values[2],
                                name='weight_sigm_{}'.format(name))

    # combinator params used in combinator calculations
    return [w_u_lin, w_z_lin, w_zu_lin, w_u_sigm, w_z_sigm,
            w_zu_sigm, w_sigm, b_lin, b_sigm]


def _curiousAI_params(shape, name):
    values = np.zeros((8,) + shape, dtype=theano.config.floatX)

    b_mu_sig = theano.shared(values[0],
                                  name='b_mu_sig_{}'.format(name))
    b_mu_lin = theano.shared(values[1],
                                  name='b_mu_lin_{}'.format(name))
    b_v_sig = theano.shared(values[2],
                                 name='b_v_sig_{}'.format(name))
    b_v_lin = theano.shared(values[3],
                                 name='b_v_lin_{}'.format(name))

    w_mu_lin = theano.shared(values[4],
                                  name='w_mu_lin_{}'.format(name))
    w_v_lin = theano.shared(values[5],
                                  name='w_v_lin_{}'.format(name))
    w_mu = theano.shared(values[6],
                              name='w_mu_{}'.format(name))
    w_v = theano.shared(values[7],
                             name='w_v_{}'.format(name))

    values = np.ones((2,) + shape, dtype=theano.config.floatX)
    w_mu_sig = theano.shared(values[0],
                                  name='w_mu_sig_{}'.format(name))
    w_v_sig = theano.shared(values[1],
                                 name='w_v_sig_{}'.format(name))

    # combinator params used in combinator calculations
    return [w_mu_lin, w_v_lin, w_mu_sig, w_v_sig, w_mu, w_v,
            b_mu_lin, b_v_lin, b_mu_sig, b_v_sig]


def _get_combinator_params(combinator_type, shape, name):
    if combinator_type == 'milaUDEM':
        return _milaUDEM_params(shape, name)
    elif combinator_type == 'curiousAI':
        return _curiousAI_params(shape, name)


def _combinator_MILAudem(z, u, combinator_params, bc_pttrn):
    w_u_lin, w_z_lin, w_zu_lin, w_u_sigm, w_z_sigm, w_zu_sigm, w_sigm, \
                                b_lin, b_sigm = combinator_params

    lin_out = w_z_lin.dimshuffle(*bc_pttrn) * z + \
              w_u_lin.dimshuffle(*bc_pttrn) * u + \
              w_zu_lin.dimshuffle(*bc_pttrn) * z * u + \
              b_lin.dimshuffle(*bc_pttrn)
    sigm_pre = w_z_sigm.dimshuffle(*bc_pttrn) * z + \
               w_u_sigm.dimshuffle(*bc_pttrn) * u + \
               w_zu_sigm.dimshuffle(*bc_pttrn) * z * u + \
               b_sigm.dimshuffle(*bc_pttrn)
    sigm_out = T.nnet.sigmoid(sigm_pre)
    output = lin_out + w_sigm.dimshuffle(*bc_pttrn) * sigm_out

    return output


def _combinator_curiousAI(z, u, combinator_params, bc_pttrn):
    w_mu_lin, w_v_lin, w_mu_sig, w_v_sig, w_mu, w_v, \
    b_mu_lin, b_v_lin, b_mu_sig, b_v_sig = combinator_params

    mu_sig_pre = w_mu_sig.dimshuffle(*bc_pttrn) * u + \
                 b_mu_sig.dimshuffle(*bc_pttrn)

    mu_lin_out = w_mu_lin.dimshuffle(*bc_pttrn) * u + \
                 b_mu_lin.dimshuffle(*bc_pttrn)

    mu_u = w_mu.dimshuffle(*bc_pttrn) * T.nnet.sigmoid(mu_sig_pre) + \
           mu_lin_out

    v_sig_pre = w_v_sig.dimshuffle(*bc_pttrn) * u + \
                b_v_sig.dimshuffle(*bc_pttrn)
    v_lin_out = w_v_lin.dimshuffle(*bc_pttrn) * u + \
                b_v_lin.dimshuffle(*bc_pttrn)

    v_u = w_v * T.nnet.sigmoid(v_sig_pre) + v_lin_out

    output = (z - mu_u) * v_u + mu_u

    return output


def _combinator(z, u, combinator_type, combinator_params):
    if u.ndim == 2:
        bc_pttrn = ('x', 0)
    elif u.ndim == 4:
        bc_pttrn = ('x', 0, 'x', 'x')
    elif u.ndim == 5:
        bc_pttrn = ('x', 0, 'x', 'x', 'x')

    if combinator_type == 'milaUDEM':
        return _combinator_MILAudem(z, u, combinator_params, bc_pttrn)
    elif combinator_type == 'curiousAI':
        return _combinator_curiousAI(z, u, combinator_params, bc_pttrn)


class CombinatorLayer(MergeLayer):
    """
    """
    def __init__(self, incoming_z, incoming_u, combinator_type, **kwargs):
        super(CombinatorLayer, self).__init__(
            [incoming_z, incoming_u], **kwargs)
        self.combinator_type = combinator_type
        z_shp, u_shp = self.input_shapes

        if len(z_shp) != 2:
            raise ValueError("The input network must have a 2-dimensional "
                             "output shape: (batch_size, num_hidden)")

        if len(u_shp) != 2:
            raise ValueError("The input network must have a 2-dimensional "
                             "output shape: (batch_size, num_hidden)")

        self.combinator_params = _get_combinator_params(combinator_type, u_shp[1:],
                                                        self.name)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        z, u = inputs
        assert z.ndim == u.ndim
        return _combinator(z, u, self.combinator_type, self.combinator_params)


def build_encoder(net, num_hidden, activation, name,
                  p_drop_hidden=0., shared_net=None):
    for i, num_nodes in enumerate(num_hidden):
        dense_lname = 'enc_dense_{}'.format(i)
        nbatchn_lname = 'enc_batchn_{}_norm'.format(i)
        noise_lname = 'enc_noise_{}'.format(i)
        lbatchn_lname = 'enc_batchn_{}_learn'.format(i)

        if shared_net is None:
            # dense pars
            W = lasagne.init.GlorotUniform()
            # batchnorm pars
            beta = lasagne.init.Constant(0)
            gamma = None if activation == rectify else lasagne.init.Constant(1)
        else:
            # dense pars
            W = shared_net[dense_lname].get_params()[0]
            # batchnorm pars
            if activation==rectify:
                beta = shared_net[lbatchn_lname].get_params()[0]
                gamma = None
            else:
                beta, gamma = shared_net[lbatchn_lname].get_params()

        net[dense_lname] = DenseLayer(net.values()[-1], num_units=num_nodes, W=W,
                                      nonlinearity=linear,
                                      name='{}_{}'.format(name, dense_lname))

        shp = net[dense_lname].output_shape[1]
        zero_const = T.zeros(shp, np.float32)
        one_const = T.ones(shp, np.float32)

        # 1. batchnormalize without learning -> goes to combinator layer
        l_name = '{}_{}'.format(name, nbatchn_lname)
        net[nbatchn_lname] = BatchNormLayer(net.values()[-1], alpha=0.1,
                                            beta=None, gamma=None, name=l_name)

        if shared_net is None:
            # add noise in dirty encoder
            net[noise_lname] = GaussianNoiseLayer(net.values()[-1],
                                                  sigma=p_drop_hidden,
                                                  name='{}_{}_'.format(name,
                                                                       noise_lname))

        # 2. batchnormalize learning, alpha one in order to depenend only on the batch mean
        l_name = '{}_{}'.format(name, lbatchn_lname)
        net[lbatchn_lname] = BatchNormLayer(net.values()[-1], alpha=1.,
                                            beta=beta, gamma=gamma, name=l_name,
                                            mean=zero_const, inv_std=one_const)

        if i < len(num_hidden) - 1:
            act_name = 'enc_activation_{}'.format(i)
            net[act_name] = NonlinearityLayer(net.values()[-1],
                                              nonlinearity=activation,
                                              name='{}_{}'.format(name, act_name))

    net['enc_softmax'] = NonlinearityLayer(net.values()[-1], nonlinearity=softmax,
                                           name='{}_enc_softmax'.format(name))

    return net['enc_softmax'], net


def build_decoder(dirty_net, clean_net, num_nodes, sigma,
                  combinator_type='milaUDEM'):
    L = len(num_nodes) - 1

    # dirty_enc_dense_1 ... z_L, dirty_enc_softmax .. u_0
    z_L = dirty_net['enc_noise_{}'.format(L)]
    dirty_net['u_0'] = BatchNormLayer(dirty_net.values()[-1], beta=None,
                                      gamma=None, name='dec_batchn_softmax')

    comb_name = 'dec_combinator_0'
    dirty_net[comb_name] = CombinatorLayer(*[z_L, dirty_net['u_0']],
                                           combinator_type=combinator_type,
                                           name=comb_name)
    enc_bname = 'enc_batchn_{}_norm'.format(L)
    mu, inv_std = clean_net[enc_bname].get_params()
    bname = 'dec_batchn_0'
    dirty_net[bname] = BatchNormLayer(dirty_net.values()[-1], alpha=1.,
                                      beta=None, gamma=None, name=bname,
                                      mean=mu, inv_std=inv_std)

    for i in range(L):
        d_name = 'dec_dense_{}'.format(L-i)
        dirty_net[d_name] = DenseLayer(dirty_net.values()[-1],
                                       num_units=num_nodes[i],
                                       nonlinearity=linear, name=d_name)
        # dirty_enc_dense_L-l ... z_l, dec_dense_l ... u_l
        z_l = dirty_net['enc_noise_{}'.format(i)]
        dirty_net['u_l'] = BatchNormLayer(dirty_net.values()[-1], beta=None,
                                          gamma=None,
                                          name='dec_batchn_dense_{}'.format(L-i))

        comb_name = 'dec_combinator_{}'.format(i+1)
        dirty_net[comb_name] = CombinatorLayer(*[z_l, dirty_net['u_l']],
                                               combinator_type=combinator_type,
                                               name=comb_name)
        enc_bname = 'enc_batchn_{}_norm'.format(L-i-1)
        mu, inv_std = clean_net[enc_bname].get_params()
        bname = 'dec_batchn_{}'.format(L-i)
        dirty_net[bname] = BatchNormLayer(dirty_net.values()[-1], alpha=1.,
                                          beta=None, gamma=None, name=bname,
                                          mean=mu, inv_std=inv_std)

    d_name = 'dec_dense_{}'.format(L+1)
    dirty_net[d_name] = DenseLayer(dirty_net.values()[-1], nonlinearity=linear,
                                   num_units=num_nodes[i+1], name=d_name)
    # input ... z_0, dec_dense_L ... u_L
    z_0 = dirty_net['inp_corr']
    dirty_net['u_L'] = BatchNormLayer(dirty_net.values()[-1], beta=None, gamma=None)

    comb_name = 'dec_combinator_{}'.format(L+1)
    dirty_net[comb_name] = CombinatorLayer(*[z_0, dirty_net['u_L']], name=comb_name,
                                           combinator_type=combinator_type)

    return dirty_net


def build_model(num_encoder, num_decoder, p_drop_input, p_drop_hidden,
                activation=rectify, batch_size=None, inp_size=None,
                combinator_type='MILAudem'):
    net = OrderedDict()
    net['input'] = InputLayer((batch_size, inp_size), name='input')
    net['inp_corr'] = GaussianNoiseLayer(net['input'], sigma=p_drop_input,
                                         name='input_corr')

    # dirty encoder
    train_output_l, dirty_encoder = build_encoder(net, num_encoder, activation,
                                                  'dirty', p_drop_hidden)

    # clean encoder
    clean_net = OrderedDict(net.items()[:1])
    eval_output_l, clean_net = build_encoder(clean_net, num_encoder, activation,
                                             'clean', 0., shared_net=dirty_encoder)

    # decoders
    dirty_net = build_decoder(dirty_encoder, clean_net, num_decoder,
                                      p_drop_hidden, combinator_type)

    return [train_output_l, eval_output_l], dirty_net, clean_net


def build_cost(X, y, num_decoder, dirty_net, clean_net, output_train, lambdas):
    class_cost = T.nnet.categorical_crossentropy(T.clip(output_train, 1e-15, 1),
                                                 y).mean()
    L = len(num_decoder)

    z_clean_l = clean_net['input']
    z_dirty_l = dirty_net['dec_combinator_{}'.format(L)]

    z_clean = lasagne.layers.get_output(z_clean_l, X, deterministic=False)
    z_dirty = lasagne.layers.get_output(z_dirty_l, X, deterministic=False)

    rec_costs = [lambdas[L] * T.sqr(z_clean - z_dirty).mean()]

    for l in range(L):
        z_clean_l = clean_net['enc_batchn_{}_norm'.format(l)]
        z_dirty_l = dirty_net['dec_batchn_{}'.format(L-l-1)]

        z_clean = lasagne.layers.get_output(z_clean_l, X, deterministic=False)
        z_dirty = lasagne.layers.get_output(z_dirty_l, X, deterministic=False)

        rec_costs.append(lambdas[l] * T.sqr(z_clean - z_dirty).mean())

    # cost = class_cost + T.sum(rec_costs)
    # return cost
    return class_cost, rec_costs
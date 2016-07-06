from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import Conv2DLayer as conv
from lasagne.layers import Deconv2DLayer as deconv
from lasagne.layers import MaxPool2DLayer as pool
from lasagne.layers.special import InverseLayer as unpool
from lasagne.layers.special import BiasLayer, ScaleLayer, NonlinearityLayer
from lasagne.layers.noise import GaussianNoiseLayer
from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import rectify, linear
import lasagne

from ladder_net_layers import CombinatorLayer, SharedNormLayer
from utils import softmax

import theano.tensor as T

from collections import OrderedDict


def build_encoder(net, num_hidden, activation, name, p_drop_hidden,
                  convolution, pooling, shared_net):
    for i, num_nodes in enumerate(num_hidden):
        affine_lname = 'enc_affine_{}'.format(i)
        nbatchn_lname = 'enc_batchn_{}_norm'.format(i)
        noise_lname = 'enc_noise_{}'.format(i)
        lbatchn_lname = 'enc_batchn_{}_learn'.format(i)

        if shared_net is None:
            # affine pars
            W = lasagne.init.GlorotUniform()
            # batchnorm pars
            beta = lasagne.init.Constant(0)
            gamma = None if activation == rectify else lasagne.init.Constant(1)
        else:
            # affine weights
            W = shared_net[affine_lname].get_params()[0]
            # batchnorm pars
            beta = shared_net[lbatchn_lname + '_beta'].get_params()[0]
            gamma = None if activation==rectify else \
                shared_net[lbatchn_lname + '_gamma'].get_params()[0]

        # affine transformation: $W \hat{h}$
        if convolution:
            net[affine_lname] = conv(net.values()[-1],
                                     num_filters=num_nodes[0],
                                     filter_size=num_nodes[1],
                                     pad=num_nodes[2], stride=num_nodes[3],
                                     W=W, nonlinearity=linear,
                                     name='{}_{}_{}'.format(name,
                                         affine_lname, 'conv'))
        else:
            net[affine_lname] = DenseLayer(net.values()[-1],
                                           num_units=num_nodes,
                                           W=W, nonlinearity=linear,
                                           name='{}_{}_{}'.format(name,
                                               affine_lname, 'affine'))

        # 1. batchnormalize without learning -> goes to combinator layer
        l_name = '{}_{}'.format(name, nbatchn_lname)
        net[nbatchn_lname] = BatchNormLayer(net.values()[-1], alpha=0.1,
                                            name=l_name, beta=None,
                                            gamma=None)

        if shared_net is None:
            # for dirty encoder -> add noise
            net[noise_lname] = GaussianNoiseLayer(net.values()[-1],
                                                  sigma=p_drop_hidden,
                                                  name='{}_{}'.format(
                                                      name, noise_lname))

        # 2. scaling & offsetting batchnormalization + noise
        l_name = '{}_{}'.format(name, lbatchn_lname)
        # offset by beta
        net[lbatchn_lname + '_beta'] = BiasLayer(net.values()[-1], b=beta,
                                                 name=l_name + '_beta')

        if gamma is not None:
            # if not rectify, scale by gamma
            net[lbatchn_lname + '_gamma'] = ScaleLayer(net.values()[-1],
                                                       scales=gamma,
                                                       name=l_name+'_gamma')

        if pooling:
            pool_name = 'enc_pool_{}'.format(i)
            net[pool_name] = pool(net.values()[-1], pool_size=num_nodes[4],
                                  stride=num_nodes[5],
                                  name='{}_{}'.format(name, pool_name))

        # apply activation
        if i < len(num_hidden) - 1:
            act_name = 'enc_activation_{}'.format(i)
            net[act_name] = NonlinearityLayer(net.values()[-1],
                                              nonlinearity=activation,
                                              name='{}_{}'.format(
                                                  name, act_name))

    # classfication layer activation -> softmax
    net['enc_softmax'] = NonlinearityLayer(net.values()[-1],
                                           nonlinearity=softmax,
                                           name='{}_enc_softmax'.format(
                                               name))

    return net['enc_softmax'], net


def build_decoder(dirty_net, clean_net, num_nodes, combinator_type,
                  convolution, pooling):
    L = len(num_nodes) - 1

    # dirty_enc_affine_1 ... z_L
    z_L = dirty_net['enc_noise_{}'.format(L)]

    # batchnormalized softmax output .. u_0 without learning bn beta, gamma
    dirty_net['u_0'] = BatchNormLayer(dirty_net.values()[-1], beta=None,
                                      gamma=None, name='dec_batchn_softmax')

    if pooling:
        unpool_name = 'dec_unpool_{}'.format(L)
        dirty_net[unpool_name] = unpool(dirty_net.values()[-1],
                                        dirty_net['enc_pool_{}'.format(L)],
                                        name=unpool_name)

    # denoised latent \hat{z}_L = g(\tilde{z}_L, u_L)
    comb_name = 'dec_combinator_0'
    dirty_net[comb_name] = CombinatorLayer(z_L, dirty_net.values()[-1],
                                           combinator_type=combinator_type,
                                           name=comb_name)

    # batchnormalize denoised latent using clean encoder's bn mean/inv_std without learning
    enc_bname = 'enc_batchn_{}_norm'.format(L)
    bname = 'dec_batchn_0'

    to_stats_l = clean_net[enc_bname]
    to_norm_l = dirty_net[comb_name]
    dirty_net[bname] = SharedNormLayer(to_stats_l, to_norm_l)

    for i in range(L):
        # dirty_enc_affine_L-i ... z_l
        z_l = dirty_net['enc_noise_{}'.format(i)]

        # affine transformation
        d_name = 'dec_affine_{}'.format(L-i)
        if convolution:
            dirty_net[d_name] = deconv(dirty_net.values()[-1],
                                       num_filters=num_nodes[i][0],
                                       filter_size=num_nodes[i][1],
                                       crop=num_nodes[i][2],
                                       stride=num_nodes[i][3],
                                       nonlinearity=linear, name=d_name +
                                                                 '_conv')
        else:
            dirty_net[d_name] = DenseLayer(dirty_net.values()[-1],
                                           num_units=num_nodes[i],
                                           nonlinearity=linear,
                                           name=d_name+'_affine')

        # batchnormalization ... u_l\
        ul_name = 'u_{}'.format(i+1)
        dirty_net[ul_name] = BatchNormLayer(dirty_net.values()[-1], alpha=1.,
                                            beta=None, gamma=None,
                                            name='dec_batchn_affine_'
                                                 '{}'.format(L-i))

        if pooling:
            unpool_name = 'dec_unpool_{}'.format(L-i-1)
            dirty_net[unpool_name] = unpool(dirty_net.values()[-1],
                                            dirty_net['enc_pool_{}'
                                                      ''.format(L-i-1)],
                                            name=unpool_name)

        # denoised latent \hat{z}_L-i
        comb_name = 'dec_combinator_{}'.format(i+1)
        dirty_net[comb_name] = CombinatorLayer(z_l, dirty_net.values()[-1],
                                               combinator_type=combinator_type,
                                               name=comb_name)

        # batchnormalized latent \hat{z}_L-i^{BN}
        enc_bname = 'enc_batchn_{}_norm'.format(L-i-1)
        bname = 'dec_batchn_{}'.format(L-i)

        to_stats_l = clean_net[enc_bname]
        to_norm_l = dirty_net[comb_name]
        dirty_net[bname] = SharedNormLayer(to_stats_l, to_norm_l)

    # corrupted input ... z_0
    z_0 = dirty_net['inp_corr']

    # affine transformation
    d_name = 'dec_affine_{}'.format(L+1)
    if convolution:
        dirty_net[d_name] = deconv(dirty_net.values()[-1],
                                   num_filters=num_nodes[i+1][0],
                                   filter_size=num_nodes[i+1][1],
                                   crop=num_nodes[i+1][2],
                                   stride=num_nodes[i+1][3],
                                   nonlinearity=linear,name=d_name+'_conv')
    else:
        dirty_net[d_name] = DenseLayer(dirty_net.values()[-1],
                                       nonlinearity=linear, name=d_name,
                                       num_units=num_nodes[i+1])

    # batchnormalization ... u_L
    dirty_net['u_{}'.format(L+1)] = BatchNormLayer(dirty_net.values()[-1], alpha=1.,
                                                   beta=None, gamma=None)

    # denoised input reconstruction
    comb_name = 'dec_combinator_{}'.format(L+1)
    dirty_net[comb_name] = CombinatorLayer(z_0, dirty_net['u_{}'.format(L+1)],
                                           name=comb_name,
                                           combinator_type=combinator_type)

    return dirty_net


def build_model(num_encoder, num_decoder, p_drop_input, p_drop_hidden,
                input_shape, batch_size=None, activation=rectify,
                combinator_type='MILAudem', convolution=False,
                pooling=False):
    net = OrderedDict()
    net['input'] = InputLayer((batch_size, ) + tuple(input_shape), # inp_size),
                              name='input')
    # corrupted input
    net['inp_corr'] = GaussianNoiseLayer(net['input'], sigma=p_drop_input,
                                         name='input_corr')

    # dirty encoder
    train_output_l, dirty_encoder = build_encoder(net, num_encoder,
                                                  activation, 'dirty',
                                                  p_drop_hidden,
                                                  convolution, pooling,
                                                  None)

    # clean encoder
    clean_net = OrderedDict(net.items()[:1])
    eval_output_l, clean_net = build_encoder(clean_net, num_encoder,
                                             activation, 'clean', 0.,
                                             convolution, pooling,
                                             shared_net=dirty_encoder)

    # dirty decoder
    dirty_net = build_decoder(dirty_encoder, clean_net, num_decoder,
                              combinator_type, convolution, pooling)

    return [train_output_l, eval_output_l], dirty_net, clean_net


def get_mu_sigma_costs(hid):
    shp = hid.shape
    mu = hid.mean(0)
    sigma = T.dot(hid.T, hid) / shp[0]

    C_mu = T.sum(mu ** 2)
    C_sigma = T.diagonal(sigma - T.log(T.clip(sigma, 1e-15, 1)))
    C_sigma -=  - T.ones_like(C_sigma)
    return C_mu, C_sigma.sum() # trace(C_sigma)


def build_cost(X, y, num_decoder, dirty_net, clean_net, output_train,
               lambdas, use_extra_costs=False, alphas=None, betas=None,
               num_labeled=None, pseudo_labels=None):
    xe = T.nnet.categorical_crossentropy
    pred = T.clip(output_train, 1e-15, 1)
    N = num_labeled if num_labeled else pred.shape[0]
    class_cost = xe(pred[:N], y[:N]).mean()

    if pseudo_labels == 'soft':
        n = 0 if num_labeled else N
        class_cost += xe(pred[n:], pred[n:]).mean()
    elif pseudo_labels == 'hard':
        M = y.shape[1]
        n = 0 if num_labeled else N
        pseudo_target = T.eye(M)[pred[n:].argmax(axis=1)]
        class_cost += xe(pred[n:], pseudo_target).mean()

    L = len(num_decoder)

    # get clean and corresponding dirty latent layer output
    z_clean_l = clean_net['input']
    z_dirty_l = dirty_net['dec_combinator_{}'.format(L)]

    z_clean = lasagne.layers.get_output(z_clean_l, X, deterministic=False)
    z_dirty = lasagne.layers.get_output(z_dirty_l, X, deterministic=False)

    # squared error
    cost = lambdas[L] * T.sqr(z_clean - z_dirty).mean()
    if use_extra_costs:
        C_mu, C_sigma = get_mu_sigma_costs(z_clean)
        cost += alphas[L] * C_mu + betas[L] * C_sigma
    rec_costs = [cost]

    for l in range(L):
        z_clean_l = clean_net['enc_batchn_{}_norm'.format(l)]
        z_dirty_l = dirty_net['dec_batchn_{}'.format(L-l-1)]

        z_clean = lasagne.layers.get_output(z_clean_l, X, deterministic=False)
        z_dirty = lasagne.layers.get_output(z_dirty_l, X, deterministic=False)

        cost = lambdas[l] * T.sqr(z_clean - z_dirty).mean()
        if use_extra_costs:
            C_mu, C_sigma = get_mu_sigma_costs(z_clean)
            cost += alphas[l] * C_mu + betas[l] * C_sigma
        rec_costs.append(cost)

    return class_cost, rec_costs
from lasagne.layers import InputLayer, DenseLayer
from lasagne.layers import Conv2DLayer as conv
from lasagne.layers import Deconv2DLayer as deconv
from lasagne.layers import MaxPool2DLayer as pool
from lasagne.layers.special import InverseLayer as unpool
from lasagne.layers.special import BiasLayer, ScaleLayer, NonlinearityLayer
from lasagne.layers.noise import GaussianNoiseLayer
from lasagne.layers.normalization import BatchNormLayer, batch_norm
from lasagne.nonlinearities import rectify, linear
import lasagne

from ladder_net_layers import CombinatorLayer, SharedNormLayer
from utils import softmax, unzip

import theano.tensor as T
from collections import OrderedDict

get_items = lambda zipped: unzip(zipped)
xe = T.nnet.categorical_crossentropy

def build_encoder(net, encoder_specs, activation, name, p_drop_hidden,
                  shared_net):
    # encoder specs is a tuple of string and tuple of integers
    for i, (transform, specs) in enumerate(encoder_specs):
        if transform == 'unpool':
            specs = net.get(specs)
        # if specs have already the name of the corresponding pool layer
        update = build_enc_layer(
            net.values()[-1], name, transform, specs, activation, i,
            p_drop_hidden, shared_net
        )
        net.update(update)
        # apply activation
        if i < len(encoder_specs) - 1:
            act_name = 'enc_activation_{}'.format(i)
            net[act_name] = NonlinearityLayer(
                net.values()[-1], nonlinearity=activation,
                name='{}_{}'.format(name, act_name)
            )

    # classfication layer activation -> softmax
    net['enc_softmax'] = NonlinearityLayer(
        net.values()[-1], nonlinearity=softmax, name=name+'_enc_softmax'
    )

    return net['enc_softmax'], net


def build_enc_layer(incoming, name, transform, specs, activation, i,
                    p_drop_hidden, shared_net):
    net = OrderedDict()
    lname = 'enc_{}_{}'.format(i, transform if 'pool' in transform else 'affine')
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
        # batchnorm pars
        beta = shared_net[lbatchn_lname + '_beta'].get_params()[0]
        gamma = None if activation == rectify else \
            shared_net[lbatchn_lname + '_gamma'].get_params()[0]
        if not isinstance(shared_net[lname], (pool, unpool)):
            # affine weights
            W = shared_net[lname].get_params()[0]
        else:
            W = None

    # affine (conv/dense/deconv) or (un)pooling transformation: $W \hat{h}$
    net[lname] = get_transform_layer(
        incoming, name+'_'+lname, transform, specs, W
    )

    # 1. batchnormalize without learning -> goes to combinator layer
    layer2bn = net.values()[-1]
    l_name = '{}_{}'.format(name, nbatchn_lname)
    bn_broadcast_cond = layer2bn.output_shape[1] == 1
    if len(layer2bn.output_shape) == 4 and bn_broadcast_cond:
        ax = (0, 1, 2, 3)
    elif len(layer2bn.output_shape) == 2 and bn_broadcast_cond:
        ax = (0, 1)
    else:
        ax = 'auto'
    net[nbatchn_lname] = BatchNormLayer(
        layer2bn, axes=ax, alpha=0.1, beta=None, gamma=None, name=l_name
    )
    if shared_net is None:
        # for dirty encoder -> add noise
        net[noise_lname] = GaussianNoiseLayer(
            net.values()[-1], sigma=p_drop_hidden,
            name='{}_{}'.format(name, noise_lname)
        )

    # 2. scaling & offsetting batchnormalization + noise
    l_name = '{}_{}'.format(name, lbatchn_lname)
    # offset by beta
    net[lbatchn_lname + '_beta'] = BiasLayer(
        net.values()[-1], b=beta, name=l_name+'_beta'
    )
    if gamma is not None:
        # if not rectify, scale by gamma
        net[lbatchn_lname + '_gamma'] = ScaleLayer(
            net.values()[-1], scales=gamma, name=l_name+'_gamma'
        )

    return net


def get_transform_layer(incoming, name, transform, specs, W):
    if transform == 'conv':
        layer = conv(
            incoming, num_filters=specs[0], filter_size=specs[1],
            stride=specs[2], pad=specs[3], nonlinearity=linear, W=W, b=None,
            name=name+'_conv'
        )
    elif transform == 'dense':
        layer = DenseLayer(
            incoming, num_units=specs, nonlinearity=linear, W=W, b=None,
            name=name+'_dense'
        )
    elif transform == 'pool':
        if len(specs) == 4:
            psize, pstride = specs[1:3]
        else:
            psize, pstride = specs
        layer = pool(
            incoming, pool_size=psize, stride=pstride, name=name
        )
    elif transform == 'deconv':
        layer = deconv(
            incoming, num_filters=specs[0], filter_size=specs[1],
            stride=specs[2], crop=specs[3], nonlinearity=linear, W=W, b=None,
            name=name+'_deconv'
        )
    elif transform == 'unpool':
        pl = specs
        # print(pl.name, pl.output_shape)
        layer = unpool(incoming, pl, name=name)

    return layer


def build_dec_layer(incoming, z_l, name, transform, specs, l,
                    combinator_type, layer2stats=None, last=False):
    dirty_net = OrderedDict()

    if l > 0:
        # transformation layer: dense, deconv, unpool
        lname = 'dec_{}_{}'.format(l, transform if 'pool' in transform
                                   else 'affine')
        if transform in ['pool', 'unpool']:
            W = None
        else:
            W = lasagne.init.GlorotUniform()
        dirty_net[lname] = get_transform_layer(incoming, name+'_'+lname,
                                               transform, specs, W)
        layer2bn = dirty_net.values()[-1]
    else:
        layer2bn = incoming

    # batchnormalization ... u_l
    ul_name = 'dec_batchn_u_{}'.format(l)
    bn_broadcast_cond = layer2bn.output_shape[1] == 1
    if len(layer2bn.output_shape) == 4 and bn_broadcast_cond:
        ax = (0, 1, 2, 3)
    elif len(layer2bn.output_shape) == 2 and bn_broadcast_cond:
        ax = (0, 1)
    else:
        ax = 'auto'
    dirty_net[ul_name] = BatchNormLayer(
        layer2bn, axes=ax, alpha=1., beta=None, gamma=None,
        name=name+'_'+ul_name
    )

    # denoised latent \hat{z}_L-i
    comb_name = 'dec_combinator_{}'.format(l)
    dirty_net[comb_name] = CombinatorLayer(
        z_l, dirty_net.values()[-1], combinator_type=combinator_type,
        name=name+'_'+comb_name
    )

    if not last:
        # batchnormalized latent \hat{z}_L-i^{BN}
        layer2norm = dirty_net[comb_name]
        bname = 'dec_batchn_z_{}'.format(l)
        dirty_net[bname] = SharedNormLayer(
            layer2stats, layer2norm, name=name+'_'+bname
        )

    return dirty_net


def build_decoder(dirty_net, clean_net, name, decoder_specs, combinator_type):
    L = len(decoder_specs) - 1
    net = OrderedDict()

    # dirty_enc_affine_1 ... z_L
    z_L = dirty_net['enc_noise_{}'.format(L)]

    # batchnormalize denoised latent using clean encoder's bn mean/inv_std
    # without learning
    enc_bname = 'enc_batchn_{}_norm'.format(L)
    layer2stats = clean_net[enc_bname]

    # batchnorm and combinator
    update = build_dec_layer(
        dirty_net.values()[-1], z_L, name, 'N/A', None, 0, combinator_type,
        layer2stats
    )
    net.update(update)

    for i, (transform, specs) in enumerate(decoder_specs[:-1]):
        # dirty_enc_affine_L-i ... z_l
        z_l = dirty_net['enc_noise_{}'.format(L-i-1)]
        enc_bname = 'enc_batchn_{}_norm'.format(L-i-1)
        layer2stats = clean_net[enc_bname]

        if transform == 'unpool':
            # print(dirty_net.keys(), specs)
            specs = dirty_net.get(specs)
        update = build_dec_layer(
            net.values()[-1], z_l, name, transform, specs, i+1,
            combinator_type, layer2stats
        )
        net.update(update)

    # corrupted input ... z_0
    z_0 = dirty_net['input_corr']
    transform, specs = decoder_specs[-1]

    if transform == 'unpool':
        specs = dirty_net.get(specs)
    update = build_dec_layer(
        net.values()[-1], z_0, name, transform, specs, i+2,
        combinator_type, None, True
    )
    net.update(update)

    return net


def build_model(encoder_specs, decoder_specs, p_drop_input, p_drop_hidden,
                input_shape, batch_size=None, activation=rectify,
                combinator_type='MILAudem'):
    net = OrderedDict()
    net['input'] = InputLayer(
        (batch_size, ) + tuple(input_shape), name='input'
    )
    # corrupted input
    net['input_corr'] = GaussianNoiseLayer(
        net['input'], sigma=p_drop_input, name='input_corr'
    )

    # dirty encoder
    train_output_l, dirty_encoder = build_encoder(
        net, encoder_specs, activation, 'dirty', p_drop_hidden, None
    )

    # clean encoder
    clean_encoder = OrderedDict(net.items()[:1])
    eval_output_l, clean_net = build_encoder(
        clean_encoder, encoder_specs, activation, 'clean', 0., dirty_encoder
    )

    # dirty decoder
    dirty_decoder = build_decoder(
        dirty_encoder, clean_net, 'dirty', decoder_specs, combinator_type
    )

    return (train_output_l, eval_output_l, dirty_encoder, dirty_decoder,
            clean_encoder)


def get_mu_sigma_costs(hid):
    shp = hid.shape
    mu = hid.mean(0)
    sigma = T.dot(hid.T, hid) / shp[0]

    C_mu = T.sum(mu ** 2)
    C_sigma = T.diagonal(sigma - T.log(T.clip(sigma, 1e-15, 1)))
    C_sigma -=  - T.ones_like(C_sigma)
    return C_mu, C_sigma.sum() # trace(C_sigma)


def build_costNstats(y_onehot, output_train, output_eval, num_labeled=None,
                     pseudo_labels=None):
    pred = T.clip(output_train, 1e-15, 1)
    N = num_labeled if num_labeled else pred.shape[0]
    class_cost = xe(pred[:N], y_onehot[:N]).mean()

    if pseudo_labels == 'soft':
        n = 0 if num_labeled else N
        class_cost += xe(pred[n:], pred[n:]).mean()
    elif pseudo_labels == 'hard':
        M = y.shape[1]
        n = 0 if num_labeled else N
        pseudo_target = T.eye(M)[pred[n:].argmax(axis=1)]
        class_cost += xe(pred[n:], pseudo_target).mean()

    pred = T.argmax(output_eval[:N], axis=1)
    y = T.argmax(y_onehot[:N], axis=1)
    accuracy = T.mean(T.eq(pred, y), dtype='float32')

    return class_cost, [accuracy]


def build_rec_costs(X, clean_net, dirty_net, decoder_specs, lambdas,
                    alphas=None, betas=None, use_extra_costs=False):
    L = len(decoder_specs)

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

    dec_batchns = [x for x in dirty_net.keys() if 'dec_batchn_z' in x][::-1]

    for l, name in enumerate(dec_batchns):
        z_clean_l = clean_net['enc_batchn_{}_norm'.format(l)]
        z_dirty_l = dirty_net[name]

        z_clean = lasagne.layers.get_output(z_clean_l, X, deterministic=False)
        z_dirty = lasagne.layers.get_output(z_dirty_l, X, deterministic=False)

        cost = lambdas[l] * T.sqr(z_clean - z_dirty).mean()
        if use_extra_costs:
            C_mu, C_sigma = get_mu_sigma_costs(z_clean)
            cost += alphas[l] * C_mu + betas[l] * C_sigma

        rec_costs.append(cost)

    return rec_costs

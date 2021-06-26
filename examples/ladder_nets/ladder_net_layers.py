from lasagne.layers import MergeLayer

import theano
import theano.tensor as T

import numpy as np

def _create_milaUDEM_params(shape, name):
    values = np.zeros((6,) + shape, dtype=theano.config.floatX)

    b_lin = theano.shared(values[0], name='bias_lin_{}'.format(name))
    b_sigm = theano.shared(values[1], name='bias_sigm_{}'.format(name))

    w_u_lin = theano.shared(values[2], name='weight_u_lin_{}'.format(name))
    w_u_sigm = theano.shared(values[3], name='weight_u_sigm_{}'.format(name))
    w_zu_lin = theano.shared(values[4], name='weight_zu_lin_{}'.format(name))
    w_zu_sigm = theano.shared(values[5], name='weight_zu_sigm_{}'.format(name))

    values = np.ones((3,) + shape, dtype=theano.config.floatX)
    w_z_lin = theano.shared(values[0], name='weight_z_lin_{}'.format(name))
    w_z_sigm = theano.shared(values[1], name='weight_z_sigm_{}'.format(name))
    w_sigm = theano.shared(values[2], name='weight_sigm_{}'.format(name))

    # combinator params used in combinator calculations
    return [w_u_lin, w_z_lin, w_zu_lin, w_u_sigm, w_z_sigm,
            w_zu_sigm, w_sigm, b_lin, b_sigm]


def _create_curiousAI_params(shape, name):
    values = np.zeros((8,) + shape, dtype=theano.config.floatX)

    b_mu_sig = theano.shared(values[0], name='b_mu_sig_{}'.format(name))
    b_mu_lin = theano.shared(values[1], name='b_mu_lin_{}'.format(name))
    b_v_sig = theano.shared(values[2], name='b_v_sig_{}'.format(name))
    b_v_lin = theano.shared(values[3], name='b_v_lin_{}'.format(name))

    w_mu_lin = theano.shared(values[4], name='w_mu_lin_{}'.format(name))
    w_v_lin = theano.shared(values[5], name='w_v_lin_{}'.format(name))
    w_mu = theano.shared(values[6], name='w_mu_{}'.format(name))
    w_v = theano.shared(values[7], name='w_v_{}'.format(name))

    values = np.ones((2,) + shape, dtype=theano.config.floatX)
    w_mu_sig = theano.shared(values[0], name='w_mu_sig_{}'.format(name))
    w_v_sig = theano.shared(values[1], name='w_v_sig_{}'.format(name))

    # combinator params used in combinator calculations
    return [w_mu_lin, w_v_lin, w_mu_sig, w_v_sig, w_mu, w_v,
            b_mu_lin, b_v_lin, b_mu_sig, b_v_sig]


def _create_combinator_params(combinator_type, shape, name):
    if combinator_type == 'milaUDEM':
        return _create_milaUDEM_params(shape, name)
    elif combinator_type == 'curiousAI':
        return _create_curiousAI_params(shape, name)


def _combinator_milaUDEM(z, u, combinator_params, bc_pttrn):
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

    output = w_sigm.dimshuffle(*bc_pttrn) * sigm_out + lin_out

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
        bc_pttrn = ('x', 0, 1, 2)

    if combinator_type == 'milaUDEM':
        return _combinator_milaUDEM(z, u, combinator_params, bc_pttrn)
    elif combinator_type == 'curiousAI':
        return _combinator_curiousAI(z, u, combinator_params, bc_pttrn)


class CombinatorLayer(MergeLayer):
    """
        A layer that combines the terms from dirty and clean encoders,
        and outputs denoised variable:
            $$ \hat{z} = g(\tilde{z}, u)$$
    """
    def __init__(self, incoming_z, incoming_u, combinator_type, **kwargs):
        super(CombinatorLayer, self).__init__(
            [incoming_z, incoming_u], **kwargs)
        self.combinator_type = combinator_type
        z_shp, u_shp = self.input_shapes

        if z_shp != u_shp:
            raise ValueError("Mismatch: input shapes must be the same. "
                             "Got dirty z ({0}) of shape {1} and clean u ({"
                             "2}) of shape {3}".format(incoming_z.name, z_shp,
                                                       incoming_u.name, u_shp))

        self.combinator_params = _create_combinator_params(combinator_type,
                                                           u_shp[1:],
                                                           self.name)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        z, u = inputs
        assert z.ndim == u.ndim
        return _combinator(z, u, self.combinator_type, self.combinator_params)


class SharedNormLayer(MergeLayer):
    """
        A layer that combines the terms from dirty and clean encoders,
        and outputs denoised variable:
            $$ \hat{z} = g(\tilde{z}, u)$$
    """
    def __init__(self, incoming2stats, incoming2norm, axes='auto', epsilon=1e-4,
                 **kwargs):
        super(SharedNormLayer, self).__init__(
            [incoming2stats, incoming2norm], **kwargs)
        stats_shp, norm_shp = self.input_shapes

        if stats_shp != norm_shp:
            raise ValueError("Mismatch: input shapes must be the same. "
                             "Got dirty z ({0}) of shape {1} and clean u ({"
                             "2}) of shape {3}"
                             .format(incoming2stats.name, stats_shp,
                                     incoming2norm.name, norm_shp))

        if axes == 'auto':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(stats_shp)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        to_stats, to_norm = inputs
        assert to_stats.ndim == to_norm.ndim

        mean = to_stats.mean(self.axes, keepdims=True)
        inv_std = T.inv(T.sqrt(to_stats.var(self.axes,
                                            keepdims=True) + self.epsilon))

        return (to_norm - mean) * inv_std
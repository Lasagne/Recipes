# Author: Nicolas Granger <nicolas.granger.m@gmail.com>
#
# Implements the connectionist temporal classification loss from:
# Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006, June).
# Connectionist temporal classification: labelling unsegmented sequence data
# with recurrent neural networks. In Proceedings of the 23rd international
# conference on Machine learning (pp. 369-376). ACM.
# ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf

import numpy as np
import theano
import theano.tensor as T
from theano.tensor import discrete_dtypes, continuous_dtypes


def isneginf(x, neginf=-1e9):
    return x < neginf


def logaddexp(x, y, magnitude=20):
    x, y = T.minimum(x, y), T.maximum(x, y)
    diff = T.minimum(y - x, magnitude)
    res = x + T.log(1 + T.exp(diff))
    return T.switch((y - x) > magnitude, y, res)


def logsumexp(x, axis, keepdims=False):
    k = T.max(x, axis=axis, keepdims=True)
    res = T.log(T.sum(T.exp(x - k), axis=axis, keepdims=keepdims)) + k
    return T.switch(isneginf(k), -2e9, res)


def log_softmax(X, axis=-1, clip=None):
    k = T.max(X, axis=axis, keepdims=True)
    norm_X = X - k

    if clip is not None:
        norm_X = T.maximum(norm_X, clip)

    log_sum_exp_X = logsumexp(norm_X, axis=axis, keepdims=True)
    return norm_X - log_sum_exp_X


# Bits of the CTC algorithm ---------------------------------------------------

def insert_alternating_blanks(labels, blank_label):
    batch_size, label_size = labels.shape
    blanked_labels = T.zeros((batch_size, 2 * label_size + 1), dtype=np.int32)
    blanked_labels = T.set_subtensor(blanked_labels[:, 0::2], blank_label)
    blanked_labels = T.set_subtensor(blanked_labels[:, 1:-1:2], labels)
    return blanked_labels


def ctc_forward(log_odds, durations,
                blanked_labels, label_sizes, not_repeated):
    seqsize, batch_sz, _ = log_odds.shape
    label_size = blanked_labels.shape[1]

    def step(t, a_tm1, log_odds_,
             durations_, blanked_labels_, label_sizes_, not_repeated_):
        y_t = log_odds_[t]
        a_t = a_tm1
        a_t = T.set_subtensor(
            a_t[:, 1:],
            logaddexp(a_t[:, 1:], a_tm1[:, :-1]))
        a_t = T.set_subtensor(
            a_t[:, 2:],
            logaddexp(a_t[:, 2:], T.switch(not_repeated_, a_tm1[:, :-2], -2e9)))

        # stop after a_T(|l'|)
        mask = T.ge(t, durations_)[:, None] \
            + T.ge(T.arange(label_size)[None, :],
                   2 * label_sizes_[:, None] + 1)

        a_t = T.switch(
            isneginf(a_t) + mask, -2e9,
            a_t + y_t[T.arange(batch_sz)[:, None], blanked_labels_])
        return a_t

    alpha_init = -2e9 * T.ones((batch_sz, label_size))
    alpha_init = T.set_subtensor(alpha_init[:, 0], 0)

    alphas, _ = theano.scan(
        fn=step,
        n_steps=seqsize,
        strict=True,
        sequences=[T.arange(seqsize)],
        outputs_info=alpha_init,
        non_sequences=[log_odds, durations, blanked_labels, label_sizes,
                       not_repeated],
        name="ctc_forward")

    return alphas


def ctc_backward(log_odds, durations, blanked_labels, label_sizes, not_repeated):
    seqsize, batch_sz, _ = log_odds.shape
    label_size = blanked_labels.shape[1]

    def step(t, b_tp1, log_odds_,
             durations_, blanked_labels_, label_sizes_, not_repeated_):
        y_t = log_odds_[t]

        # increase b_{T+1}(|l'|) from 0 to 1 to initialize recursion
        starter_t = T.eq(t, durations_ - 1)[:, None] \
            * T.eq((2 * label_sizes_)[:, None],
                   T.arange(label_size)[None, :])
        b_tp1_2lp1 = b_tp1[T.arange(batch_sz), 2 * label_sizes_]
        b_tp1 = T.set_subtensor(
            b_tp1_2lp1,
            T.switch(T.eq(t, durations_ - 1), 0, b_tp1_2lp1))
        b_tp1 = T.switch(starter_t, 0, b_tp1)  # initialize recursion

        b_t = b_tp1
        b_t = T.set_subtensor(
            b_t[:, :-1],
            logaddexp(b_t[:, :-1], b_tp1[:, 1:]))
        b_t = T.set_subtensor(
            b_t[:, :-2],
            logaddexp(b_t[:, :-2], T.switch(not_repeated_, b_tp1[:, 2:], -2e9)))
        b_t += y_t[T.arange(batch_sz)[:, None], blanked_labels_]
        b_t = T.switch(isneginf(b_t), -2e9, b_t)
        return b_t

    beta_init = -2e9 * T.ones((batch_sz, label_size))

    betas, _ = theano.scan(
        fn=step,
        n_steps=seqsize,
        strict=True,
        sequences=[T.arange(seqsize)],
        outputs_info=beta_init,
        non_sequences=[log_odds, durations, blanked_labels, label_sizes,
                       not_repeated],
        go_backwards=True,
        name="ctc_backward")
    betas = betas[::-1, :, :]

    return betas


# Theano Op -------------------------------------------------------------------

def ctc_propagate(linout, durations, blanked_labels, label_sizes, not_repeated):
    _, batch_size, voca_size = linout.shape

    logits = log_softmax(linout)
    betas = ctc_backward(logits, durations,
                         blanked_labels, label_sizes, not_repeated)
    loss = - logaddexp(betas[0, :, 0], betas[0, :, 1])

    # alphas = ctc_forward(logits, durations,
    #                      blanked_labels, label_sizes, not_repeated)
    # loss = - logaddexp(
    #     alphas[durations - 1, T.arange(batch_size), 2 * label_sizes - 1],
    #     alphas[durations - 1, T.arange(batch_size), 2 * label_sizes])

    return loss, logits, betas


def ctc_backprop(durations, blanked_labels, label_sizes, not_repeated,
                 logits, betas, loss, output_gradient):
    seq_size, batch_size, voca_size = logits.shape

    alphas = ctc_forward(logits, durations,
                         blanked_labels, label_sizes, not_repeated)

    # log(sum_{s \in lab(l, k)} a_t(s) b_t(s))
    def fwbw_sum_step(k, s, labels_, ab_):
        s_view = s[:, T.arange(batch_size), labels_[:, k]]
        ab_view = ab_[:, :, k]
        next_sum = logaddexp(s_view, ab_view)
        s = T.set_subtensor(s_view, next_sum)
        return s

    ab = alphas + betas
    fwbw_sum = theano.scan(
        fn=fwbw_sum_step,
        sequences=[T.arange(blanked_labels.shape[1])],
        outputs_info=-2e9 * T.ones((seq_size, batch_size, voca_size)),
        non_sequences=[blanked_labels, ab],
        strict=True,
        name="fwbw_sum")[0][-1]  # should be unrolled if label_size is known

    A = loss[None, :, None] + logits \
        + logsumexp(fwbw_sum - logits, axis=2, keepdims=True)
    B = loss[None, :, None] + fwbw_sum - logits
    dloss_dy = T.exp(A) - T.exp(B)

    dloss_dy = T.switch(T.all(isneginf(fwbw_sum), axis=2, keepdims=True),
                        0, dloss_dy)

    return dloss_dy * output_gradient[None, :, None]


def make_ctc_op():
    preds_var = T.tensor3()
    durations_var = T.ivector()
    blanked_labels_var = T.imatrix()
    bool_matrix = T.TensorType("bool", (False, False))
    not_repeated_var = bool_matrix()
    label_sizes_var = T.ivector()

    # linout, durations, labels, label_sizes, blank = inputs
    # seq_size, batch_size, voca_size = linout.shape
    #
    # logits, blanked_labels, not_repeated, betas, loss = \
    #     ctc_perform_graph(linout, durations, labels, label_sizes, blank)

    loss, logits, betas = ctc_propagate(preds_var, durations_var, blanked_labels_var,
                                        label_sizes_var, not_repeated_var)

    def backprop_op1(inputs, output_gradients):
        del inputs
        return [
            output_gradients[0],
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type()]

    op1 = theano.OpFromGraph(
        inputs=[preds_var, durations_var,
                blanked_labels_var, label_sizes_var,
                not_repeated_var],
        outputs=[preds_var, logits, betas, loss],
        grad_overrides=backprop_op1,
        inline=True, name="ctcLossOp1")

    def backprop_op2(inputs, output_gradients):
        preds_var_, logits_, betas_, loss_, \
            durations_, blanked_labels_, label_sizes_, not_repeated_ = inputs
        output_gradient, = output_gradients

        g = ctc_backprop(durations_, blanked_labels_, label_sizes_, not_repeated_,
                         logits_, betas_, loss_, output_gradient)

        return [
            g,
            T.zeros_like(logits_),
            # theano.gradient.disconnected_type(),
            T.zeros_like(betas_),
            # theano.gradient.disconnected_type(),
            T.zeros_like(loss_),
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type()]

    preds, logits, betas, loss = op1(
        preds_var, durations_var,
        blanked_labels_var, label_sizes_var,
        not_repeated_var)

    op2 = theano.OpFromGraph(
        inputs=[preds, logits, betas, loss,
                durations_var, blanked_labels_var, label_sizes_var,
                not_repeated_var],
        outputs=[loss + preds.sum() * 0 + logits.sum() * 0 + betas.sum() * 0],
        grad_overrides=backprop_op2,
        inline=True, name="ctcLossOp2")

    return op1, op2


# -----------------------------------------------------------------------------

def ctc_loss(preds, durations, labels, label_sizes, blank=-1):
    """Compute the Connectionnist Temporal Classification loss [#graves2006]_.

    .. math:: L = - ln\left( \sum_{\pi \in \mathcal{B}^{-1}(l)} P(\pi | y)
                      \right)

    where :math:`y` is the sequence of predictions, :math:`l` the target
    label sequence without blanks or repetition, :math:`\pi` is taken from the
    ensemble of possible label assignments over the observations and
    :math:`\mathcal{B}` is a function that remove blanks and repetitions for a
    sequence of labels.

    Parameters
    ----------
    preds : Theano shared variable, expression or numpy array
        The input values for the softmax function with shape
        duration x batch_size x nclasses.
    durations: Theano shared variable, expression or numpy array
        An _integer_ vector of size batch_size contining the actual length of
        each sequence in preds.
    labels: Theano shared variable, expression or numpy array
        An _integer_ matrix of size batch_size x label_size containing the
        target labels.
    label_sizes: Theano shared variable, expression or numpy array
        An _integer_ vector of size batch_size containing the actual length of
        each sequence in labels.
    blank:
        The blank label class, by default the last index.

    Returns
    -------
    Theano tensor
        A vector expression with the CTC loss of each sequence.

    Reference
    ---------
    .. [#graves2006] Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J.
       (2006, June). Connectionist temporal classification: labelling
       unsegmented sequence data with recurrent neural networks. In
       Proceedings of the 23rd international conference on Machine learning
       (pp. 369-376). ACM. ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
    """
    preds = T.as_tensor_variable(preds)
    durations = T.as_tensor_variable(durations)
    labels = T.as_tensor_variable(labels)
    label_sizes = T.as_tensor_variable(label_sizes)
    blank = T.cast(T.as_tensor_variable(blank), 'int32')

    if not(preds.dtype in continuous_dtypes and preds.ndim == 3):
        raise ValueError("preds must continuous with dimension 3")
    if not (durations.dtype in discrete_dtypes and durations.ndim == 1):
        raise ValueError("durations must be a integer vector")
    if not (labels.dtype in discrete_dtypes and labels.ndim == 2):
        raise ValueError("labels must be an integer matrix")
    if not (label_sizes.dtype in discrete_dtypes and label_sizes.ndim == 1):
        raise ValueError("label_sizes must be an integer vector")
    if not (blank.dtype in discrete_dtypes and blank.ndim == 0):
        raise ValueError("blank must be an integer value")

    voca_size = T.cast(preds.shape[2], 'int32')
    labels = labels % voca_size
    blank = blank % voca_size

    op1, op2 = make_ctc_op()

    blanked_labels = insert_alternating_blanks(labels, blank)
    not_repeated = T.neq(blanked_labels[:, 2:], blanked_labels[:, :-2])

    preds, logits, betas, loss = op1(preds, durations,
                                     blanked_labels, label_sizes,
                                     not_repeated)
    loss = op2(preds, logits, betas, loss,
               durations, blanked_labels, label_sizes, not_repeated)

    return loss

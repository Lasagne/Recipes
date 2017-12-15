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


# Bits of the CTC algorithm ---------------------------------------------------

def insert_alternating_blanks(labels, blank_label):
    batch_size, label_size = labels.shape
    blanked_labels = T.zeros((batch_size, 2 * label_size + 1), dtype=np.int32)
    blanked_labels = T.set_subtensor(blanked_labels[:, 0::2], blank_label)
    blanked_labels = T.set_subtensor(blanked_labels[:, 1:-1:2], labels)
    return blanked_labels


def ctc_forward(log_odds, seq_sizes,
                blanked_labels, label_sizes, not_repeated):
    batch_dur, batch_sz, _ = log_odds.shape
    batch_dur, batch_sz = T.cast(batch_dur, 'int32'), T.cast(batch_sz, 'int32')
    label_size = blanked_labels.shape[1]

    def step(t, a_tm1, log_odds_,
             seq_sizes_, blanked_labels_, label_sizes_, not_repeated_):
        y_t = log_odds_[t]
        k = T.max(a_tm1, axis=-1, keepdims=True)
        k = T.switch(T.all(T.isinf(a_tm1), axis=-1, keepdims=True), 0, k)
        a_tm1 = T.switch(T.isinf(a_tm1), 0, T.exp(a_tm1 - k))  # exit log space
        a_t = a_tm1
        a_t = T.inc_subtensor(a_t[:, 1:], a_tm1[:, :-1])
        a_t = T.inc_subtensor(a_t[:, 2:], a_tm1[:, :-2] * not_repeated_)

        # stop after a_T(|l'|)
        mask = T.ge(t, seq_sizes_)[:, None] \
            + T.ge(T.arange(label_size)[None, :],
                   2 * label_sizes_[:, None] + 1)

        a_t = T.switch(  # back to log space
            T.eq(a_t, 0) + mask, -np.inf,
            T.log(a_t) + k + y_t[T.arange(batch_sz)[:, None], blanked_labels_])
        return a_t

    alpha_init = -np.inf * T.ones((batch_sz, label_size))
    alpha_init = T.set_subtensor(alpha_init[:, 0], 0)

    alphas, _ = theano.scan(
        fn=step,
        sequences=[T.arange(batch_dur)],
        outputs_info=[alpha_init],
        non_sequences=[log_odds, seq_sizes, blanked_labels, label_sizes,
                       not_repeated],
        name="ctc_forward")

    return alphas


def ctc_backward(log_odds, seq_sizes,
                 blanked_labels, label_sizes, not_repeated):
    batch_dur, batch_sz, _ = log_odds.shape
    label_size = blanked_labels.shape[1]

    def step(t, b_tp1, log_odds_,
             seq_sizes_, blanked_labels_, label_sizes_, not_repeated_):
        y_t = log_odds_[t]
        k = T.max(b_tp1, axis=-1, keepdims=True)
        k = T.switch(T.all(T.isinf(b_tp1), axis=-1, keepdims=True), 0, k)
        b_tp1 = T.switch(T.isinf(b_tp1), 0, T.exp(b_tp1 - k))  # exit log space

        # increase b_{T+1}(|l'|) from 0 to 1 to initialize recursion
        starter_t = T.eq(t, seq_sizes_ - 1)[:, None] \
            * T.eq((2 * label_sizes_)[:, None],
                   T.arange(label_size)[None, :]) * 1
        b_tp1 += starter_t  # initialize recursion

        b_t = b_tp1
        b_t = T.inc_subtensor(b_t[:, :-1], b_tp1[:, 1:])
        b_t = T.inc_subtensor(b_t[:, :-2], b_tp1[:, 2:] * not_repeated_)
        b_t = T.switch(  # back to log space
            T.eq(b_t, 0), -np.inf,
            T.log(b_t) + k + y_t[T.arange(batch_sz)[:, None], blanked_labels_])
        return b_t

    beta_init = - np.inf * T.ones((batch_sz, label_size))

    betas, _ = theano.scan(
        fn=step,
        sequences=[T.arange(batch_dur)],
        outputs_info=[beta_init],
        non_sequences=[log_odds, seq_sizes, blanked_labels, label_sizes,
                       not_repeated],
        go_backwards=True,
        name="ctc_backward")
    betas = betas[::-1, :, :]

    return betas


# Theano Op -------------------------------------------------------------------

def ctc_perform_graph(preds, seq_sizes, labels, label_sizes, blank):
    _, batch_size, voca_size = preds.shape

    log_preds = T.log(preds)
    blanked_labels = insert_alternating_blanks(labels, blank)
    not_repeated = T.neq(blanked_labels[:, 2:], blanked_labels[:, :-2])
    betas = ctc_backward(log_preds, seq_sizes,
                         blanked_labels, label_sizes, not_repeated)

    loss = -T.switch(T.all(T.isinf(betas[0, :, :2]), axis=1),
                     -np.inf,  # impossible sequences, eg: too short
                     T.log(T.exp(betas[0, :, 0]) + T.exp(betas[0, :, 1])))

    return log_preds, blanked_labels, not_repeated, betas, loss


def ctc_grad_graph(inputs, output_gradients):
    linear_out, seq_durations, labels, label_sizes, _ = inputs
    seq_size, batch_size, voca_size = linear_out.shape
    label_size = labels.shape[1]

    # TODO: will theano optimize this redundant call when both loss and
    # gradient are requested separately?
    log_preds, blanked_labels, not_repeated, betas, loss = \
        ctc_perform_graph(*inputs)

    alphas = ctc_forward(log_preds, seq_durations,
                         blanked_labels, label_sizes, not_repeated)

    log_pl = - loss

    # sum_{s \in lab(l, k)} a_t(s) b_t(s)
    def fwbw_sum_step(k, s, labels_, ab_):
        s_view = s[:, T.arange(batch_size), labels_[:, k]]
        ab_view = ab_[:, :, k]
        next_sum = ab_view + T.switch(T.isinf(s_view),
                                      0, T.log(1 + T.exp(s_view - ab_view)))
        next_sum = T.switch(T.isinf(ab_view), s_view, next_sum)
        s = T.set_subtensor(s_view, next_sum)
        return s

    ab = T.switch(T.isinf(alphas) + T.isinf(betas), -np.inf, alphas + betas)
    fwbw_sum = theano.scan(
        fn=fwbw_sum_step,
        sequences=[T.arange(2 * label_size + 1)],
        outputs_info=[-np.inf * T.ones((seq_size, batch_size, voca_size))],
        non_sequences=[blanked_labels, ab],
        name="fwbw_sum")[0][-1]

    # d(loss) / dy
    dloss_dy = T.switch(
        T.isinf(loss)[None, :, None],
        0,
        - T.exp(fwbw_sum - log_pl[None, :, None] - 2 * log_preds))

    return [dloss_dy * output_gradients[0][None, :, None],
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type(),
            theano.gradient.disconnected_type()]


def make_ctc_op():
    linear_out_var = T.tensor3()
    seq_durations_var = T.ivector()
    labels_var = T.imatrix()
    label_sizes_var = T.ivector()
    blank_var = T.iscalar()

    _, _, _, _, loss = ctc_perform_graph(
        linear_out_var, seq_durations_var, labels_var,
        label_sizes_var, blank_var)

    return theano.OpFromGraph(
        inputs=[linear_out_var, seq_durations_var,
                labels_var, label_sizes_var, blank_var],
        outputs=[loss],
        grad_overrides=ctc_grad_graph,
        inline=True, name="ctcLossOp")


CTCLossOp = make_ctc_op()


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
        The probabilities of each class (for example the output of a softmax
        function) with shape duration x batch_size x nclasses.
    durations: Theano shared variable, expression or numpy array
        An _integer_ vector of size batch_size contining the actual length of
        each sequence in preds.
    labels: Theano shared variable, expression or numpy array
        An _integer_ matrix of size batch_size x label_size containg the target
        labels.
    label_sizes: Theano shared variable, expression or numpy array
        An _integer_ vector of size batch_size contining the actual length of
        each sequence in labels.
    blank:
        The blank label class, by default the last one.

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

    return CTCLossOp(preds, durations, labels, label_sizes, blank)

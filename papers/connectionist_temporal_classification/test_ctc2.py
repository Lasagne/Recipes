import unittest
import numpy as np
import theano
import theano.tensor as T
from theano.tests import unittest_tools

from papers.connectionist_temporal_classification.ctc import \
    ctc_loss, ctc_forward, ctc_backward, insert_alternating_blanks, isneginf


def log_softmax(X):
    k = T.max(X, axis=-1, keepdims=True)
    norm_X = X - k
    log_sum_exp_X = T.log(T.sum(T.exp(norm_X), axis=-1, keepdims=True))
    return norm_X - log_sum_exp_X


class TestCTC(unittest.TestCase):
    def setUp(self):
        unittest_tools.seed_rng()

    def test_forward_backward(self):
        batch_size = 6
        label_size = 7
        voca_size = 5
        seq_size = 10

        label_lengths = np.random.randint(0, label_size,
                                          size=(batch_size,), dtype=np.int32)
        label_lengths[0] = label_size  # extremum case
        label_lengths[1] = 0  # extremum case
        labels = np.array(
            [np.random.randint(0, voca_size - 1, size=label_size, dtype=np.int32)
             for _ in range(batch_size)])
        for i in range(batch_size):
            labels[i, label_lengths[i]:] = -1

        seq_durations = np.array([
            np.random.randint(max(1, label_lengths[i]), seq_size)
            for i in range(batch_size)], dtype=np.int32)

        linear_out = np.random.randn(seq_size, batch_size, voca_size) \
            .astype(np.float32)

        blank_class = -1
        blank_class = np.mod(blank_class, voca_size)

        labels = np.mod(labels, voca_size)

        log_odds = log_softmax(linear_out)
        blanked_labels = insert_alternating_blanks(T.mod(labels, voca_size),
                                                   blank_class)
        not_repeated = T.neq(blanked_labels[:, 2:], blanked_labels[:, :-2])

        alphas = ctc_forward(log_odds, seq_durations,
                             blanked_labels, label_lengths, not_repeated)
        betas = ctc_backward(log_odds, seq_durations,
                             blanked_labels, label_lengths, not_repeated)

        preds = log_softmax(linear_out)

        y_blanks = preds[:, T.arange(batch_size)[:, None], blanked_labels]
        p_l = T.sum(T.exp(alphas + betas - y_blanks), axis=2)

        alphas = alphas.eval()
        betas = betas.eval()
        preds = preds.eval()

        for i in range(batch_size):
            assert np.allclose(alphas[0, i, 0], preds[0, i, -1])
            if label_lengths[i] > 0:
                assert np.allclose(alphas[0, i, 1], preds[0, i, labels[i, 0]])
            else:
                assert isneginf(alphas[0, i, 1])
            assert np.all(isneginf(alphas[0, i, 2:]))

        for i in range(batch_size):
            t = seq_durations[i] - 1
            l = label_lengths[i]
            assert np.allclose(betas[t, i, 2 * l], preds[t, i, -1])
            if l > 0:
                assert np.allclose(betas[t, i, 2 * l - 1],
                                   preds[t, i, labels[i, l - 1]])
                assert np.all(isneginf(betas[t, i, :max(l - 2, 0)]))
            else:
                assert np.all(isneginf(betas[t, i, 1:]))

        p_l = p_l.eval()

        for i in range(batch_size):
            assert (np.allclose(p_l[:seq_durations[i], i], p_l[0, i]))
            a, b = max(0, 2 * label_lengths[i] - 1), 2 * label_lengths[i] + 1
            p_li = np.exp(alphas[seq_durations[i] - 1, i, a:b]).sum()
            assert np.allclose(p_li, p_l[0, i])
            p_li = np.exp(betas[0, i, :2]).sum()
            assert np.allclose(p_li, p_l[0, i])

    def test_simple_precomputed(self):
        # Test obtained from Torch tutorial at:
        # https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md

        linear_out = np.asarray([
            [[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [-5, -4, -3, -2, -1]],
            [[0, 0, 0, 0, 0], [6, 7, 8, 9, 10], [-10, -9, -8, -7, -6]],
            [[0, 0, 0, 0, 0], [11, 12, 13, 14, 15], [-15, -14, -13, -12, -11]]
        ], dtype=np.float32)

        seq_sizes = np.asarray([1, 3, 3], dtype=np.int32)

        labels = np.asarray([[1, 0], [3, 3], [2, 3]], dtype=np.int32)

        label_sizes = np.asarray([1, 2, 2], dtype=np.int32)

        expected_losses = np.asarray([1.609437943, 7.355742931, 4.938849926],
                                     dtype=np.float32)

        blank = 0

        expected_grad = np.asarray([
            [[0.2,            -0.8,            0.2,            0.2,           0.2],
             [ 0.01165623125,  0.03168492019,  0.08612854034, -0.7658783197,  0.636408627],
             [-0.02115798369,  0.03168492019, -0.8810571432,   0.2341216654,  0.636408627]],
            [[0,               0,              0,              0,             0],
             [-0.9883437753,   0.03168492019,  0.08612854034,  0.2341216654,  0.636408627],
             [-0.02115798369,  0.03168492019, -0.1891518533,  -0.4577836394,  0.636408627]],
            [[0,               0,              0,              0,             0],
             [0.01165623125,   0.03168492019,  0.08612854034, -0.7658783197,  0.636408627],
             [-0.02115798369,  0.03168492019,  0.08612854034, -0.7330639958,  0.636408627]]
        ], dtype=np.float32)

        linear_out_var = T.as_tensor_variable(linear_out)
        losses = ctc_loss(
            linear_out_var, seq_sizes, labels, label_sizes, blank)

        assert np.allclose(losses.eval(), expected_losses, atol=1)

        grad = theano.grad(losses.sum(), wrt=linear_out_var)

        assert np.allclose(grad.eval(), expected_grad, rtol=.001, atol=1)

    def test_random(self):
        batch_size = 16
        label_size = 5
        voca_size = 4
        seq_size = 20

        label_sizes = np.random.randint(
            0, label_size, size=(batch_size,), dtype=np.int32)
        label_sizes[0] = label_size
        label_sizes[1] = 0
        label_sizes[2] = 5
        label_sizes[3] = 5

        labels = np.random.randint(
            0, voca_size - 1,
            size=(batch_size, label_size), dtype=np.int32)
        labels[3] = 0

        seq_sizes = np.array([
            np.random.randint(max(1, label_sizes[i]), seq_size)
            for i in range(batch_size)], dtype=np.int32)
        seq_sizes[2] = 4

        linear_out = np.random.randn(
            seq_size, batch_size, voca_size).astype(np.float32)

        # check edge cases
        # TODO

        # check the gradient can be computed at all
        linear_out_var = T.tensor3()
        preds = T.nnet.softmax(
            linear_out_var.reshape((-1, voca_size))
        ).reshape((seq_size, batch_size, voca_size))

        g = theano.grad(ctc_loss(preds, seq_sizes,
                                 labels, label_sizes).sum(),
                        wrt=linear_out_var).eval(
            {linear_out_var: linear_out.astype(np.float32)})
        assert not np.any(np.isnan(g))

        # check correctness against finite difference approximation
        def f(linear_out_):
            preds_ = T.nnet.softmax(
                    linear_out_.reshape((-1, voca_size))
                ).reshape((seq_size, batch_size, voca_size))
            loss = ctc_loss(preds_, seq_sizes, labels, label_sizes)
            # prevent finite differences from failing
            loss = T.switch(isneginf(-loss), 0, loss)
            return loss

        unittest_tools.verify_grad(
            f, [linear_out], rel_tol=0.1, abs_tol=1)

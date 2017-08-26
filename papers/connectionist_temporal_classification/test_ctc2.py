import unittest
import numpy as np
import theano
import theano.tensor as T
from theano.tests import unittest_tools

from ctc import ctc_loss


class TestCTC(unittest.TestCase):
    def setUp(self):
        unittest_tools.seed_rng()

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
            [[0.2,            -0.8,          0.2,            0.2, 0.2],
             [0.01165623125, 0.03168492019, 0.08612854034, -0.7658783197,
              0.636408627],
             [-0.02115798369, 0.03168492019, -0.8810571432, 0.2341216654,
              0.636408627]],
            [[0, 0, 0, 0, 0],
             [-0.9883437753, 0.03168492019, 0.08612854034, 0.2341216654,
              0.636408627],
             [-0.02115798369, 0.03168492019, -0.1891518533, -0.4577836394,
              0.636408627]],
            [[0, 0, 0, 0, 0],
             [0.01165623125, 0.03168492019, 0.08612854034, -0.7658783197,
              0.636408627],
             [-0.02115798369, 0.03168492019, 0.08612854034, -0.7330639958,
              0.636408627]]
        ], dtype=np.float32)

        seq_size, batch_size, voca_size = linear_out.shape

        linear_out_t = T.as_tensor_variable(linear_out)
        seq_sizes_t = T.as_tensor_variable(seq_sizes)
        labels_t = T.as_tensor_variable(labels)
        label_sizes_t = T.as_tensor_variable(label_sizes)
        blank_t = T.as_tensor_variable(blank)

        preds = T.nnet.softmax(
            linear_out_t.reshape((-1, voca_size))
        ).reshape((seq_size, batch_size, voca_size))
        losses = ctc_loss(preds, seq_sizes_t, labels_t, label_sizes_t, blank_t)

        assert np.allclose(losses.eval(), expected_losses)

        grad = theano.grad(losses.sum(), wrt=linear_out_t)

        assert np.allclose(grad.eval(), expected_grad)

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
                        wrt=linear_out_var).eval({linear_out_var: linear_out})
        assert not np.any(np.isnan(g))

        # check correctness against finite difference approximation
        def f(linear_out_):
            preds_ = T.nnet.softmax(
                    linear_out_.reshape((-1, voca_size))
                ).reshape((seq_size, batch_size, voca_size))
            loss = ctc_loss(preds_, seq_sizes, labels, label_sizes)
            # prevent finite differences from failing
            loss = T.switch(T.isinf(loss), 0, loss)
            return loss

        unittest_tools.verify_grad(
            f, [linear_out], rel_tol=0.1)

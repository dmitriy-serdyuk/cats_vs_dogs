__author__ = 'dima'

import numpy

from theano import tensor
from theano import function

from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks import Rectifier, Softmax

from cats_vs_dogs.bricks import ConvNN, Convolutional, Pooling


def test_convolutional():
    x = tensor.tensor4('x')
    n_channels = 4
    conv = Convolutional((3, 3), 3, n_channels, (1, 1),
                         weights_init=Constant(1.))
    conv.initialize()
    y = conv.apply(x)
    func = function([x], y)

    x_val = numpy.ones((5, n_channels, 17, 13))
    assert numpy.all(func(x_val) == 3 * 3 * n_channels * numpy.ones((15, 11)))


def test_pooling():
    x = tensor.tensor4('x')
    n_channels = 4
    batch_size = 5
    x_size = 17
    y_size = 13
    pool_size = 3
    pool = Pooling((pool_size, pool_size))
    y = pool.apply(x)
    func = function([x], y)

    x_val = numpy.ones((batch_size, n_channels, x_size, y_size))
    assert numpy.all(func(x_val) == numpy.ones((batch_size, n_channels,
                                                x_size / pool_size + 1,
                                                y_size / pool_size + 1)))


def test_conv_nn():
    # We test only dimension
    conv_mlp = ConvNN([Rectifier(), Rectifier()], (3, 200, 200),
                      [(10, 3, 4, 4), (20, 10, 5, 5)], [(7, 7), (7, 7)],
                      [Softmax()], [2],
                      weights_init=IsotropicGaussian(0.1),
                      biases_init=Constant(0.))
    conv_mlp.initialize()
    x = tensor.tensor4('x')
    out = conv_mlp.apply(x)
    func = function([x], out)
    x_val = numpy.ones((35, 3, 200, 200))
    assert func(x_val).shape == (35, 2)

__author__ = 'dima'

import numpy

from theano import tensor
from theano import function

from blocks.initialization import Constant

from cats_vs_dogs.bricks import ConvMLP, Convolutional, Pooling


def test_convolutional():
    x = tensor.tensor4('x')
    n_channels = 4
    conv = Convolutional(3, 3, n_channels, weights_init=Constant(1.))
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
    pool = Pooling(pool_size)
    y = pool.apply(x)
    func = function([x], y)

    x_val = numpy.ones((batch_size, n_channels, x_size, y_size))
    assert numpy.all(func(x_val) == numpy.ones((batch_size, n_channels,
                                                x_size / pool_size + 1,
                                                y_size / pool_size + 1)))

__author__ = 'dima'

import numpy
from collections import OrderedDict

import theano
from theano import tensor

from blocks.algorithms import StepRule
from blocks.utils import shared_floatx_zeros


def shared_scalar(value):
    return theano.shared(numpy.cast[theano.config.floatX](value))


class Adam(StepRule):
    """A step in the direction opposite to the gradient.

    Parameters
    ----------
    learning_rate : float
        The learning rate by which the gradient is multiplied to produce
        the descent step.

    Attributes
    ----------
    learning_rate : :class:`~tensor.TensorSharedVariable`
        The shared variable storing the learning rate used.

    """
    def __init__(self, learning_rate=0.0002, b1=0.1, b2=0.001,
                 e=1e-8):
        self.learning_rate = shared_scalar(learning_rate)
        self.i = shared_scalar(0.)
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.updates = OrderedDict()

    def compute_steps(self, gradients):
        i_t = self.i + 1.
        fix1 = 1. - (1. - self.b1)**i_t
        fix2 = 1. - (1. - self.b2)**i_t
        lr_t = self.learning_rate * (tensor.sqrt(fix2) / fix1)
        param_updates = OrderedDict()
        for param, gradient in gradients.iteritems():
            m = shared_floatx_zeros(param.get_value().shape)
            v = shared_floatx_zeros(param.get_value().shape)
            m_t = (self.b1 * gradient) + ((1. - self.b1) * m)
            v_t = (self.b2 * tensor.sqr(gradient)) + ((1. - self.b2) * v)
            g_t = m_t / (tensor.sqrt(v_t) + self.e)

            param_step = -lr_t * g_t
            self.updates[m] = m_t
            self.updates[v] = v_t
            param_updates[param] = param_step
        self.updates[self.i] = i_t
        return param_updates

    def additional_updates(self):
        return self.updates

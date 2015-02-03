__author__ = 'serdyuk'

from itertools import chain

from theano.tensor.nnet.conv import conv2d

from blocks.bricks import WEIGHTS, Sequence, Initializable, Feedforward
from blocks.bricks import lazy
from blocks.bricks.base import application
from blocks.roles import add_role, BIASES
from blocks.utils import shared_floatx_zeros


class Convolutional(Initializable, Feedforward):
    @lazy
    def __init__(self, conv_size, step, border_mode='valid', **kwargs):
        super(Convolutional, self).__init__(**kwargs)
        self.conv_size = conv_size
        self.step = step
        self.border_mode = border_mode

    def _allocate(self):
        W = shared_floatx_zeros((self.conv_size, self.conv_size), name='W')
        add_role(W, WEIGHTS)
        self.params.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')

    def _initialize(self):
        W, = self.params
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input plus optional bias

        """
        W, = self.params
        output = conv2d(input_, W, border_mode=self.border_mode)
        return output


class ConvMLP(Sequence, Initializable, Feedforward):
    @lazy
    def __init__(self, activations, dims, **kwargs):
        self.activations = activations

        self.conv_transformations = [Convolutional(name='linear_{}'.format(i))
                                     for i in range(len(activations))]
        # Interleave the transformations and activations
        application_methods = [brick.apply for brick in list(chain(*zip(
            self.conv_transformations, activations))) if brick is not None]
        if not dims:
            dims = [None] * (len(activations) + 1)
        self.dims = dims
        super(ConvMLP, self).__init__(application_methods, **kwargs)

    @property
    def input_dim(self):
        return self.dims[0]

    @input_dim.setter
    def input_dim(self, value):
        self.dims[0] = value

    @property
    def output_dim(self):
        return self.dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.dims) - 1 == len(self.linear_transformations):
            raise ValueError
        for input_dim, output_dim, layer in zip(self.dims[:-1], self.dims[1:],
                                                self.linear_transformations):
            layer.input_dim = input_dim
            layer.output_dim = output_dim
            layer.use_bias = self.use_bias

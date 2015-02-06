__author__ = 'serdyuk'

import math
from itertools import chain

import numpy

from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

from blocks.bricks import WEIGHTS, Sequence, Initializable, Feedforward, Brick
from blocks.bricks import lazy, MLP
from blocks.bricks.base import application
from blocks.roles import add_role, BIASES
from blocks.utils import shared_floatx_zeros


class Convolutional(Initializable, Feedforward):
    """Convolutional layer.

    Parameters
    ----------
    conv_size : a tuple (kernel x size, kernel y size)
    num_featuremaps : number of feature maps
    num_channels : number of input channels
    step : a tuple (convolution step x, convolution step y)
    border_mode : border mode 'valid' or 'full'
    """
    @lazy
    def __init__(self, conv_size, num_featuremaps, num_channels, step,
                 border_mode='valid', **kwargs):
        super(Convolutional, self).__init__(**kwargs)
        self.conv_size = conv_size
        self.border_mode = border_mode
        self.num_featuremaps = num_featuremaps
        self.num_channels = num_channels
        self.step = step

    def _allocate(self):
        conv_size_x, conv_size_y = self.conv_size
        W = shared_floatx_zeros((self.num_featuremaps, self.num_channels,
                                 conv_size_x, conv_size_y), name='W')
        add_role(W, WEIGHTS)
        self.params.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')

    def _initialize(self):
        W, = self.params
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the convolutional transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input

        """
        W, = self.params
        output = conv2d(input_, W, subsample=self.step,
                        border_mode=self.border_mode)
        return output


class Pooling(Initializable, Feedforward):
    """Pooling layer.

    Parameters
    ----------
    pooling_size : a tuple (pooling size x, pooling size y)
    """
    @lazy
    def __init__(self, pooling_size, **kwargs):
        super(Pooling, self).__init__(**kwargs)
        self.pooling_size = pooling_size

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the pooling (subsampling) transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input

        """
        output = max_pool_2d(input_, self.pooling_size)
        return output


class Flattener(Brick):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        batch_size = input_.shape[0]
        return input_.reshape((batch_size, -1))


class ConvNN(Sequence, Initializable, Feedforward):
    @lazy
    def __init__(self, conv_activations, input_dim, conv_dims, pooling_dims,
                 top_mlp_activations, top_mlp_dims, conv_steps=None, **kwargs):
        if conv_steps == None:
            self.conv_steps = (1, 1)
        self.conv_activations = conv_activations
        self.input_dim = input_dim
        self.conv_dims = conv_dims
        self.pooling_dims = pooling_dims
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims

        self.conv_transformations = [Convolutional(name='conv_{}'.format(i))
                                     for i in range(len(conv_activations))]
        self.subsamplings = [Pooling(name='pooling_{}'.format(i))
                             for i in range(len(conv_activations))]
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)
        # Interleave the transformations and activations
        application_methods = [brick.apply for brick in list(chain(*zip(
            self.conv_transformations, self.subsamplings, conv_activations)))
            if brick is not None]
        self.flattener = Flattener()
        if len(top_mlp_activations) > 0:
            application_methods += [self.flattener.apply]
            application_methods += [self.top_mlp.apply]
        super(ConvNN, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        if not len(self.conv_dims) == len(self.pooling_dims):
            raise ValueError('Dimension mismatch')
        inp_conv_dims = [self.input_dim] + self.pooling_dims[:-1]
        layer_list = zip(inp_conv_dims, self.conv_dims, self.pooling_dims,
                         self.conv_transformations, self.subsamplings)
        curr_output_dim = self.input_dim
        for conv_inp_dim, conv_out_dim, pool_dim, conv, pool in layer_list:
            num_featuremaps, channels, size_x, size_y = conv_out_dim
            conv.conv_size = (size_x, size_y)
            conv.num_featuremaps = num_featuremaps
            conv.num_channels = channels
            conv.step = self.conv_steps

            pool.pooling_size = pool_dim

            _, curr_x, curr_y = curr_output_dim
            # TODO: consider case of full convolution, step != 1 and ignore
            # border
            curr_output_dim = (num_featuremaps,
                               math.ceil((curr_x - size_x + 1) /
                                         float(pool_dim[0])),
                               math.ceil((curr_y - size_y + 1) /
                                         float(pool_dim[1])))

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(curr_output_dim)] + self.top_mlp_dims

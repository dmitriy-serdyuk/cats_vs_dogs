__author__ = 'serdyuk'

from itertools import chain

import numpy

from theano import tensor
from theano.tensor.nnet.conv import conv2d, ConvOp
from theano.tensor.signal.downsample import max_pool_2d, DownsampleFactorMax

from blocks.bricks import WEIGHTS, Sequence, Initializable, Feedforward, Brick
from blocks.bricks import lazy, MLP
from blocks.bricks.base import application
from blocks.roles import add_role, BIASES
from blocks.utils import shared_floatx_zeros


class Convolutional(Initializable):
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
    def __init__(self, conv_size, num_channels, step,
                 border_mode='valid', **kwargs):
        super(Convolutional, self).__init__(**kwargs)
        self.conv_size = conv_size
        self.border_mode = border_mode
        self.num_channels = num_channels
        self.step = step

    def _allocate(self):
        num_featuremaps, conv_size_x, conv_size_y = self.conv_size
        W = shared_floatx_zeros((num_featuremaps, self.num_channels,
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


class Pooling(Initializable):
    """Pooling layer.

    Parameters
    ----------
    pooling_size : a tuple (pooling size x, pooling size y)
    """
    @lazy
    def __init__(self, pooling_size, ignore_border=False, **kwargs):
        super(Pooling, self).__init__(**kwargs)
        self.pooling_size = pooling_size
        self.ignore_border = ignore_border

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
        output = max_pool_2d(input_, self.pooling_size, self.ignore_border)
        return output


class Flattener(Brick):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        batch_size = input_.shape[0]
        return input_.reshape((batch_size, -1))


class ContrastNormalization(Brick):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        axises = tensor.arange(1, input_.ndim)
        means = input_.mean(axis=axises)
        output = input_ - means.dimshuffle(0, None * input_.ndim)
        return output


class ConvPool(Sequence, Initializable, Feedforward):
    @lazy
    def __init__(self, input_dim, conv_size, conv_step, pool_size,
                 conv_type='valid', ignore_border=False, **kwargs):
        self.input_dim = input_dim
        self.conv_size = conv_size
        self.pool_size = pool_size
        self.conv_type = conv_type
        self.ignore_border = ignore_border
        self.conv_step = conv_step

        if conv_size is not None:
            num_featuremaps, conv_size = conv_size[0], conv_size[1:]
        else:
            num_featuremaps, conv_size = None, None
        if input_dim is not None:
            num_channels, _, _ = input_dim
        else:
            num_channels = None
        self.convolution = Convolutional(conv_size, num_featuremaps,
                                         num_channels, conv_step, name='conv')
        self.pooling = Pooling(pool_size, name='pooling')
        application_methods = [self.convolution.apply, self.pooling.apply]
        super(ConvPool, self).__init__(application_methods, **kwargs)


    @property
    def output_dim(self):
        inp_dim = numpy.array(self.input_dim[1:])
        num_featuremaps = self.conv_size[0]
        conv_dim = numpy.array(self.conv_size[1:])

        out_dim = ConvOp.getOutputShape(inp_dim, conv_dim, self.conv_step,
                                        self.conv_type)
        out_dim = DownsampleFactorMax.out_shape(out_dim, self.pool_size,
                                                self.ignore_border)

        return num_featuremaps, out_dim[0], out_dim[1]

    def push_allocation_config(self):
        self.convolution.conv_size = self.conv_size
        self.convolution.border_mode = self.conv_type
        self.convolution.num_channels = self.input_dim[0]
        self.convolution.step = self.conv_step

        self.pooling.pooling_size = self.pool_size
        self.pooling.ignore_border = self.ignore_border


class ConvNN(Sequence, Initializable, Feedforward):
    @lazy
    def __init__(self, conv_activations, input_dim, conv_dims, pooling_dims,
                 top_mlp_activations, top_mlp_dims, conv_step=None, **kwargs):
        if conv_step == None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.conv_activations = conv_activations
        self.input_dim = input_dim
        self.conv_dims = conv_dims
        self.pooling_dims = pooling_dims
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims

        self.transformations = [ConvPool(name='conv_pool_{}'.format(i))
                                for i in range(len(conv_activations))]
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)
        # Interleave the transformations and activations
        application_methods = [brick.apply for brick in list(chain(*zip(
            self.transformations, conv_activations)))
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
        inp_conv_dims = [self.input_dim] + self.conv_dims[:-1]
        layer_list = zip(inp_conv_dims, self.conv_dims, self.pooling_dims,
                         self.transformations)
        curr_output_dim = self.input_dim
        for conv_inp_dim, conv_dim, pool_dim, layer in layer_list:
            num_channels, _, _ = conv_inp_dim
            layer.input_dim = curr_output_dim
            layer.conv_size = conv_dim
            layer.conv_step = self.conv_step
            layer.pool_size = pool_dim

            curr_output_dim = layer.output_dim

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(curr_output_dim)] + self.top_mlp_dims

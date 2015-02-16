__author__ = 'serdyuk'

from itertools import chain

import numpy

from theano import tensor

from blocks.bricks import Sequence, Initializable, Feedforward, Brick
from blocks.bricks import lazy, MLP
from blocks.bricks.base import application
from blocks.bricks.conv import ConvolutionalLayer, Flattener


class ContrastNormalization(Brick):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        axises = tensor.arange(1, input_.ndim)
        means = input_.mean(axis=axises)
        output = input_ - means.dimshuffle(0, None * input_.ndim)
        return output


class ConvNN(Sequence, Initializable, Feedforward):
    def __init__(self, conv_activations, input_dim, filter_sizes,
                 feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims, conv_step=None, **kwargs):
        if conv_step == None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.input_dim = input_dim
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims

        params = zip(conv_activations, filter_sizes, feature_maps,
                     pooling_sizes)
        self.layers = [ConvolutionalLayer(filter_size=filter_size,
                                          num_filters=num_filter,
                                          num_channels=None,
                                          pooling_size=pooling_size,
                                          activation=activation.apply,
                                          conv_step=self.conv_step,
                                          name='conv_pool_{}'.format(i))
                       for i, (activation, filter_size, num_filter,
                               pooling_size)
                       in enumerate(params)]
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)
        # Interleave the transformations and activations
        application_methods = [brick.apply for brick in list(chain(*zip(
            self.layers)))
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
        curr_output_dim = self.input_dim
        for layer in self.layers:
            num_channels, _, _ = curr_output_dim
            layer.convolution.num_channels = num_channels
            layer.convolution.image_shape = curr_output_dim[1:]
            layer.pooling.input_dim = layer.convolution.get_dim('output')

            curr_output_dim = layer.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(curr_output_dim)] + self.top_mlp_dims

import re

import numpy

import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams

from blocks.bricks import Sequence, Initializable, Feedforward, Brick
from blocks.bricks import MLP
from blocks.bricks.base import application
from blocks.bricks.conv import (ConvolutionalLayer, Flattener,
                                ConvolutionalSequence)
from blocks.filter import VariableFilter
from blocks.roles import INPUT, WEIGHT
from blocks.graph import ComputationGraph

floatX = theano.config.floatX


class ContrastNormalization(Brick):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        axises = tensor.arange(1, input_.ndim)
        means = input_.mean(axis=axises)
        output = input_ - means.dimshuffle(0, None * input_.ndim)
        return output


class ConvNN(Sequence, Initializable, Feedforward):
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims, conv_step=None,
                 border_mode='valid', **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        params = zip(conv_activations, filter_sizes, feature_maps,
                     pooling_sizes)
        self.layers = [ConvolutionalLayer(filter_size=filter_size,
                                          num_filters=num_filter,
                                          pooling_size=pooling_size,
                                          activation=activation.apply,
                                          conv_step=self.conv_step,
                                          border_mode=self.border_mode,
                                          name='conv_pool_{}'.format(i))
                       for i, (activation, filter_size, num_filter,
                               pooling_size)
                       in enumerate(params)]
        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        application_methods = [self.conv_sequence.apply]
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)
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
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims


class Dropout(object):
    def __init__(self, prob, inputs, outputs):
        self.graph = ComputationGraph(outputs)
        self.inputs = inputs
        self.outputs = outputs
        self.prob = prob

    def train_model(self):
        srng = RandomStreams(seed=876)
        input_vars = VariableFilter(roles=[INPUT])(self.graph)
        replacements = {var: var * srng.binomial(var.shape, p=self.prob,
                                                 dtype=floatX)
                        for var in input_vars if re.match('linear', var.name)}
        new_graph = self.graph.replace(replacements)
        out_names = [o.name for o in self.outputs]
        new_outputs = [var for var in new_graph.outputs
                       if var.name in out_names]
        return new_outputs

    def test_model(self):
        weight_vars = VariableFilter(roles=[WEIGHT])(self.graph)
        replacements = {var: var * self.prob for var in weight_vars
                        if re.match('linear', var.name)}
        new_graph = self.graph.replace(replacements)
        out_names = [o.name for o in self.outputs]
        new_outputs = [var for var in new_graph.outputs
                       if var.name in out_names]
        return new_outputs


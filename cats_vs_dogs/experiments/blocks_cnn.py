import logging
import os
import argparse
import yaml

import numpy

import theano
from theano import tensor

from blocks.algorithms import (GradientDescent, Scale, CompositeRule,
                               StepClipping, RMSProp)
from blocks.bricks import Softmax, Rectifier
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.config_parser import Configuration
from blocks.extensions import FinishAfter, Printing, Timing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import LoadFromDump, Dump
from blocks.extensions.training import SharedVariableModifier
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation
from blocks.roles import INPUT, WEIGHT

import fuel
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, MultiProcessing

from cats_vs_dogs.iterators import (DogsVsCats, UnbatchStream,
                                    RandomCrop, Reshape,
                                    ImageTranspose, OneHotEncoderStream,
                                    RandomRotate, Normalize)
from cats_vs_dogs.bricks import ConvNN
from cats_vs_dogs.algorithms import Adam
from cats_vs_dogs.schemes import SequentialShuffledScheme

floatX = theano.config.floatX
logging.basicConfig(level='INFO')


def parse_config(path):
    config = ConfigCats()
    config.add_config('image_shape', type_=int, default=221)
    config.add_config('scaled_size', type_=int, default=256)
    config.add_config('channels', type_=int, default=3)
    config.add_config('batch_size', type_=int, default=100)
    config.add_config('epochs', type_=int, default=50000)
    config.add_config('load', type_=bool, default=False)
    config.add_config('model_path', type_=str, default='./models/model')
    config.add_config('algorithm', type_=str, default=False)
    config.add_config('feature_maps', type_=list, default=[25, 50, 100])
    config.add_config('conv_sizes', type_=list, default=[7, 5, 3])
    config.add_config('pool_sizes', type_=list, default=[3, 3, 3])
    config.add_config('mlp_hiddens', type_=list, default=[500])
    config.add_config('learning_rate', type_=float, default=1.e-4)
    config.add_config('dropout', type_=bool, default=False)
    config.add_config('plot', type_=bool, default=False)
    config.add_config('rotate', type_=bool, default=True)
    config.add_config('usel2', type_=bool, default=False)
    config.add_config('l2regularization', type_=float, default=0.01)
    config.add_config('max_store', type_=int, default=5)
    config.load_yaml(path)
    return config


class ConfigCats(Configuration):
    def load_yaml(self, path):
        yaml_file = os.path.expanduser(path)
        if os.path.isfile(yaml_file):
            with open(yaml_file) as f:
                for key, value in yaml.safe_load(f).items():
                    if key not in self.config:
                        raise ValueError("Unrecognized config in YAML: {}"
                                         .format(key))
                    self.config[key]['yaml'] = value


def construct_stream(dataset, config, train=False):
    rng = numpy.random.RandomState(9682)
    stream = DataStream(
        dataset=dataset,
        iteration_scheme=SequentialShuffledScheme(dataset.num_examples,
                                                  config.batch_size, rng))
    stream = UnbatchStream(data_stream=stream)

    stream = Reshape(data_stream=stream, image_source='X',
                     shape_source='shape')
    if config.rotate and train:
        crop_size = (config.image_shape + config.scaled_size) / 2.
    else:
        crop_size = config.image_shape
    stream = RandomCrop(data_stream=stream,
                        crop_size=crop_size,
                        scaled_size=config.scaled_size,
                        image_source='X',
                        rng=rng)
    if config.rotate and train:
        stream = RandomRotate(data_stream=stream,
                              input_size=crop_size,
                              output_size=config.image_shape,
                              image_source='X',
                              rng=rng)
    stream = RandomCrop(data_stream=stream,
                        crop_size=config.image_shape,
                        scaled_size=config.scaled_size,
                        image_source='X',
                        rng=rng)
    stream = Normalize(data_stream=stream, image_source='X')
    stream = Batch(
        data_stream=stream,
        iteration_scheme=ConstantScheme(config.batch_size))
    stream = OneHotEncoderStream(num_classes=2, data_stream=stream,
                                 target_source='y')
    stream = ImageTranspose(data_stream=stream, image_source='X')

    return stream


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def main(**kwargs):
    config = AttrDict(kwargs)

    conv_activations = [Rectifier() for _ in config.feature_maps]
    mlp_activations = [Rectifier() for _ in config.mlp_hiddens] + [None]
    convnet = ConvNN(conv_activations, config.channels,
                     (config.image_shape,) * 2,
                     filter_sizes=zip(config.conv_sizes, config.conv_sizes),
                     feature_maps=config.feature_maps,
                     pooling_sizes=zip(config.pool_sizes, config.pool_sizes),
                     top_mlp_activations=mlp_activations,
                     top_mlp_dims=config.mlp_hiddens + [2],
                     border_mode='full',
                     weights_init=IsotropicGaussian(0.1),
                     biases_init=Constant(0))
    convnet.initialize()
    for layer in convnet.layers:
        logging.info('layer dim: (%d, %d, %d)' % layer.get_dim('input_'))
    logging.info('layer dim: (%d, %d, %d)' % layer.get_dim('output'))

    x = tensor.tensor4('X')
    y = tensor.lmatrix('y')
    last_hidden = convnet.apply(x)
    cost = Softmax().categorical_cross_entropy(y, last_hidden)
    cost.name = 'cost'
    error_rate = MisclassificationRate().apply(tensor.argmax(y, axis=1),
                                               last_hidden)
    error_rate.name = 'error_rate'

    ouputs = [cost, error_rate]
    train_outputs = ouputs
    test_outputs = ouputs
    cg = ComputationGraph(cost)
    if config.dropout:
        last_inputs = VariableFilter(bricks=[convnet.top_mlp],
                                     roles=[INPUT])(cg.variables)
        cg_dropout = apply_dropout(cg, last_inputs, 0.5)
        train_outputs = cg_dropout.outputs
    if config.usel2:
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        train_outputs[0] = (cost + config.l2regularization *
                            sum([(weight ** 2).sum() for weight in weights]))
        test_outputs[0] = (cost + config.l2regularization *
                           sum([(weight ** 2).sum() for weight in weights]))

    logging.info('.. model built')
    train_dataset = DogsVsCats('train', os.path.join(fuel.config.data_path,
                                                     'dogs_vs_cats',
                                                     'train.h5'))
    train_stream = construct_stream(train_dataset, config, train=True)
    test_dataset = DogsVsCats('test', os.path.join(fuel.config.data_path,
                                                   'dogs_vs_cats',
                                                   'train.h5'))
    test_stream = construct_stream(test_dataset, config)
    valid_dataset = DogsVsCats('valid', os.path.join(fuel.config.data_path,
                                                     'dogs_vs_cats',
                                                     'train.h5'))
    valid_stream = construct_stream(valid_dataset, config)

    valid_monitor = DataStreamMonitoring(
        variables=test_outputs, data_stream=valid_stream, prefix="valid")
    test_monitor = DataStreamMonitoring(
        variables=test_outputs, data_stream=test_stream, prefix="test")

    extensions = [ProgressBar()]
    if config.load:
        extensions.append(LoadFromDump(config.model_path))

    if config.algorithm == 'adam':
        step_rule = Adam(config.learning_rate)
    elif config.algorithm == 'rms_prop':
        step_rule = RMSProp(config.learning_rate)
    else:
        clipping = StepClipping(threshold=numpy.cast[floatX](100.))
        sgd = Scale(learning_rate=config.learning_rate)
        step_rule = CompositeRule([clipping, sgd])
        adjust_learning_rate = SharedVariableModifier(
            sgd.learning_rate,
            lambda n: 10. / (10. / config.learning_rate + n))
        extensions.append(adjust_learning_rate)
    algorithm = GradientDescent(cost=train_outputs[0], step_rule=step_rule,
                                params=cg.parameters)
    train_monitor = TrainingDataMonitoring(
        variables=train_outputs + [
            aggregation.mean(algorithm.total_gradient_norm)],
        prefix="train", after_epoch=True)
    extensions.extend([FinishAfter(after_n_epochs=config.epochs),
                       train_monitor,
                       valid_monitor,
                       test_monitor,
                       Printing(),
                       Dump(config.model_path, after_epoch=True,
                            before_first_epoch=True)])
    if config.plot:
        extensions.extend([Plot(os.path.basename(config.model_path),
                                [[train_monitor.record_name(cost),
                                  valid_monitor.record_name(cost),
                                  test_monitor.record_name(cost)],
                                 [train_monitor.record_name(error_rate),
                                  valid_monitor.record_name(error_rate),
                                  test_monitor.record_name(error_rate)]],
                                every_n_batches=20)])

    extensions.append(Timing())
    model = Model(train_outputs[0])
    main_loop = MainLoop(model=model, data_stream=train_stream,
                         algorithm=algorithm, extensions=extensions)
    main_loop.run()


if __name__ == '__main__':
    logging.info('.. starting')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()
    config = parse_config(args.config)

    main(**{name: getattr(config, name) for name in config.config})

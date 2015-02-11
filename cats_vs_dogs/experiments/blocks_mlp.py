import logging
import os
import argparse
import yaml

import numpy

import theano
from theano import tensor

from blocks.bricks import MLP, Tanh, Softmax, Rectifier
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from blocks.datasets import DataStream
from blocks.datasets.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.algorithms import (GradientDescent, SteepestDescent, CompositeRule,
                               GradientClipping)
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import SerializeMainLoop, LoadFromDump, Dump
from blocks.config_parser import Configuration

from ift6266h15.code.pylearn2.datasets.variable_image_dataset import RandomCrop

from cats_vs_dogs.iterators import DogsVsCats
from cats_vs_dogs.bricks import Convolutional, Pooling, ConvNN
from cats_vs_dogs.algorithms import Adam
from cats_vs_dogs.schemes import SequentialShuffledScheme
from cats_vs_dogs.extentions import (DumpWeights, LoadWeights,
                                     AdjustParameter)

floatX = theano.config.floatX
logging.basicConfig(level='INFO')


def parse_config(path):
    config = ConfigCats()
    config.add_config('image_shape', type_=int,
                        default=221)
    config.add_config('scaled_size', type_=int,
                        default=256)
    config.add_config('channels', type_=int,
                        default=3)
    config.add_config('batch_size', type_=int,
                        default=100)
    config.add_config('epochs', type_=int,
                        default=50000)
    config.add_config('load', type_=bool,
                        default=False)
    config.add_config('model_path', type_=str,
                        default='./models/model')
    config.add_config('use_adam', type_=bool,
                        default=False)
    config.add_config('feature_maps', type_=list,
                        default=[25, 50, 100])
    config.add_config('conv_sizes', type_=list,
                        default=[7, 5, 3])
    config.add_config('pool_sizes', type_=list,
                        default=[3, 3, 3])
    config.add_config('mlp_hiddens', type_=list,
                        default=[500])
    config.add_config('learning_rate', type_=float,
                        default=1.e-4)
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


if __name__ == '__main__':
    logging.info('.. starting')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()
    config = parse_config(args.config)

    input_dim = (config.channels, config.image_shape, config.image_shape)
    mlp_activations = [Rectifier() for _ in config.mlp_hiddens] + [Softmax()]
    model = ConvNN([Rectifier(), Rectifier()], input_dim,
                   zip(config.feature_maps, config.conv_sizes, config.conv_sizes),
                   zip(config.pool_sizes, config.pool_sizes),
                   mlp_activations, config.mlp_hiddens + [2],
                   weights_init=IsotropicGaussian(0.1),
                   biases_init=Constant(0.))
    model.initialize()

    x = tensor.tensor4('X')
    y = tensor.lmatrix('y')
    y_hat = model.apply(x)
    cost = CategoricalCrossEntropy().apply(y, y_hat)
    error_rate = MisclassificationRate().apply(y[:, 0], y_hat)

    logging.info('.. model built')
    rng = numpy.random.RandomState(2014 + 02 + 04)
    transformer = RandomCrop(config.scaled_size, config.image_shape, rng)
    train_dataset = DogsVsCats('train', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'),
                               transformer)
    train_stream = DataStream(
        dataset=train_dataset,
        iteration_scheme=SequentialShuffledScheme(train_dataset.num_examples,
                                          config.batch_size, rng))
    test_dataset = DogsVsCats('test', os.path.join('${PYLEARN2_DATA_PATH}',
                                                   'dogs_vs_cats',
                                                   'train.h5'),
                              transformer)
    test_stream = DataStream(
        dataset=test_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples,
                                          config.batch_size))
    valid_dataset = DogsVsCats('valid', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'),
                               transformer)
    valid_stream = DataStream(
        dataset=valid_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples,
                                          config.batch_size))

    train_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=train_stream, prefix="train")
    valid_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=valid_stream, prefix="valid")
    test_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=test_stream, prefix="test")

    extensions = []
    if config.load:
        extensions += [LoadWeights(config.model_path)]
    extensions += [FinishAfter(after_n_epochs=config.epochs),
                   #train_monitor,
                   #valid_monitor,
                   #test_monitor,
                   SerializeMainLoop('./models/main.pkl'),
                   Printing(),
                   Dump(config.model_path, after_every_epoch=True,
                               before_first_epoch=True)]

    if config.use_adam:
        step_rule = Adam()
    else:
        clipping = GradientClipping(threshold=numpy.cast[floatX](1000.))
        sgd = SteepestDescent(learning_rate=config.learning_rate)
        step_rule = CompositeRule([clipping, sgd])
        adjust_learning_rate = AdjustParameter(sgd.learning_rate,
                                                  lambda n: 100. / (100. + n))
        extensions += [adjust_learning_rate]
    main_loop = MainLoop(
        model, data_stream=train_stream,
        algorithm=GradientDescent(
            cost=cost, step_rule=step_rule),
        extensions=extensions)
    main_loop.run()
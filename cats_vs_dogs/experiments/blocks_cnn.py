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
from blocks.datasets import DataStream
from blocks.datasets.schemes import ConstantScheme
from blocks.datasets.streams import BatchDataStream
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import SerializeMainLoop, LoadFromDump, Dump
from blocks.extensions.training import SharedVariableModifier
from blocks.config_parser import Configuration

from cats_vs_dogs.iterators import (DogsVsCats, UnbatchStream,
                                    RandomCropStream, ReshapeStream,
                                    ImageTransposeStream, OneHotEncoderStream)
from cats_vs_dogs.bricks import ConvNN, Dropout
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


def construct_stream(dataset, config):
    stream = DataStream(
        dataset=dataset,
        iteration_scheme=SequentialShuffledScheme(dataset.num_examples,
                                                  config.batch_size, rng))
    stream = OneHotEncoderStream(num_classes=2, data_stream=stream)
    stream = UnbatchStream(data_stream=stream)
    stream = ReshapeStream(data_stream=stream)
    stream = RandomCropStream(data_stream=stream,
                              crop_size=config.image_shape,
                              scaled_size=config.scaled_size, rng=rng)
    stream = BatchDataStream(data_stream=stream,
                             iteration_scheme=ConstantScheme(config.batch_size)
    )
    stream = ImageTransposeStream(data_stream=stream)
    return stream


if __name__ == '__main__':
    logging.info('.. starting')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()
    config = parse_config(args.config)

    conv_activations = [Rectifier() for _ in config.feature_maps]
    mlp_activations = ([Rectifier() for _ in config.mlp_hiddens] +
                       [Softmax().categorical_cross_entropy])
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

    x = tensor.tensor4('X')
    y = tensor.lmatrix('y')
    y_hat = convnet.apply(x)
    cost = CategoricalCrossEntropy().apply(y, y_hat)
    error_rate = MisclassificationRate().apply(tensor.argmax(y, axis=1), y_hat)

    ouputs = [cost, error_rate]
    train_outputs = ouputs
    test_outputs = ouputs
    if config.dropout:
        dropout = Dropout(0.5, [x, y], ouputs)
        train_outputs = dropout.train_model()
        test_outputs = dropout.test_model()

    logging.info('.. model built')
    rng = numpy.random.RandomState(2014 + 02 + 04)
    train_dataset = DogsVsCats('train', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'))
    train_stream = construct_stream(train_dataset, config)
    test_dataset = DogsVsCats('test', os.path.join('${PYLEARN2_DATA_PATH}',
                                                   'dogs_vs_cats',
                                                   'train.h5'))
    test_stream = construct_stream(test_dataset, config)
    valid_dataset = DogsVsCats('valid', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'))
    valid_stream = construct_stream(valid_dataset, config)

    valid_monitor = DataStreamMonitoring(
        variables=test_outputs, data_stream=valid_stream, prefix="valid")
    test_monitor = DataStreamMonitoring(
        variables=test_outputs, data_stream=test_stream, prefix="test")

    extensions = []
    if config.load:
        extensions += [LoadFromDump(config.model_path)]

    if config.algorithm == 'adam':
        step_rule = Adam()
    elif config.algorithm == 'rms_prop':
        step_rule = RMSProp(config.learning_rate)
    else:
        clipping = StepClipping(threshold=numpy.cast[floatX](100.))
        sgd = Scale(learning_rate=config.learning_rate)
        step_rule = CompositeRule([clipping, sgd])
        adjust_learning_rate = SharedVariableModifier(
            sgd.learning_rate,
            lambda n: 10. / (10. / config.learning_rate + n))
        extensions += [adjust_learning_rate]
    algorithm = GradientDescent(cost=train_outputs[0], step_rule=step_rule)
    train_monitor = TrainingDataMonitoring(
        variables=train_outputs + [
                   aggregation.mean(algorithm.total_gradient_norm)],
        prefix="train", after_every_epoch=True)
    extensions += [FinishAfter(after_n_epochs=config.epochs),
                   train_monitor,
                   valid_monitor,
                   test_monitor,
                   Printing(),
                   Dump(config.model_path, after_every_epoch=True,
                        before_first_epoch=True)]
    model = Model(train_outputs[0])
    main_loop = MainLoop(model=model, data_stream=train_stream,
                         algorithm=algorithm, extensions=extensions)
    main_loop.run()
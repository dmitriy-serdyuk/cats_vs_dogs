import logging
import os
import argparse

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
from cats_vs_dogs.extentions import DumpWeights, LoadWeights

from ift6266h15.code.pylearn2.datasets.variable_image_dataset import RandomCrop

from cats_vs_dogs.iterators import DogsVsCats
from cats_vs_dogs.bricks import Convolutional, Pooling, ConvNN
from cats_vs_dogs.algorithms import Adam
from cats_vs_dogs.schemes import SequentialShuffledScheme

floatX = theano.config.floatX
logging.basicConfig(level='INFO')


def parse_args():
    parser = argparse.ArgumentParser("Cats vs Dogs training algorithm")
    parser.add_argument('--image-shape', type=int,
                        default=221,
                        help='Image shape')
    parser.add_argument('--scaled-size', type=int,
                        default=256,
                        help='Scaled size')
    parser.add_argument('--channels', type=int,
                        default=3,
                        help='Number of channels')
    parser.add_argument('--batch-size', type=int,
                        default=100,
                        help='Batch size')
    parser.add_argument('--epochs', type=int,
                        default=50000,
                        help='Number of epochs')
    parser.add_argument('--load', action='store_true',
                        default=False,
                        help='Load the parameters')
    parser.add_argument('--model-path',
                        default='./models/model')
    parser.add_argument('--use-adam', action='store_true',
                        default=False,
                        help='Use Adam optimizer')
    parser.add_argument('--feature-maps', nargs='+', type=int,
                        default=[25, 50, 100],
                        help='Number of feature maps for layers')
    parser.add_argument('--conv-sizes', nargs='+', type=int,
                        default=[7, 5, 3],
                        help='Convolution sizes for layers')
    parser.add_argument('--pool-sizes', nargs='+', type=int,
                        default=[3, 3, 3],
                        help='Pooling sizes for layers')
    parser.add_argument('--mlp-hiddens', nargs='+', type=int,
                        default=[500],
                        help='Number of hidden units in full connected layers')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.info('.. starting')
    input_dim = (args.channels, args.image_shape, args.image_shape)
    model = ConvNN([Rectifier(), Rectifier()], input_dim,
                   zip(args.feature_maps, args.conv_sizes, args.conv_sizes),
                   zip(args.pool_sizes, args.pool_sizes),
                   [Rectifier(), Softmax()], args.mlp_hiddens + [2],
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
    transformer = RandomCrop(args.scaled_size, args.image_shape, rng)
    train_dataset = DogsVsCats('train', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'),
                               transformer)
    train_stream = DataStream(
        dataset=train_dataset,
        iteration_scheme=SequentialShuffledScheme(train_dataset.num_examples,
                                          args.batch_size, rng))
    test_dataset = DogsVsCats('test', os.path.join('${PYLEARN2_DATA_PATH}',
                                                   'dogs_vs_cats',
                                                   'train.h5'),
                              transformer)
    test_stream = DataStream(
        dataset=test_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples,
                                          args.batch_size))
    valid_dataset = DogsVsCats('valid', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'),
                               transformer)
    valid_stream = DataStream(
        dataset=valid_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples,
                                          args.batch_size))

    train_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=train_stream, prefix="train")
    valid_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=valid_stream, prefix="valid")
    test_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=test_stream, prefix="test")

    extensions = []
    if args.load:
        extensions += [LoadWeights(args.model_path)]
    extensions += [FinishAfter(after_n_epochs=args.epochs),
                   train_monitor,
                   valid_monitor,
                   test_monitor,
                   SerializeMainLoop('./models/main.pkl'),
                   Printing(),
                   DumpWeights(args.model_path, after_every_epoch=True,
                               before_first_epoch=True)]

    if args.use_adam:
        step_rule = Adam()
    else:
        step_rule = CompositeRule([GradientClipping(threshold=numpy.cast[floatX](1000.)),
                                   SteepestDescent(learning_rate=1.e-5)])
    main_loop = MainLoop(
        model, data_stream=train_stream,
        algorithm=GradientDescent(
            cost=cost, step_rule=step_rule),
        extensions=extensions)
    main_loop.run()
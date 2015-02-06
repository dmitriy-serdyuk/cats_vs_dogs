import logging
import os
import argparse

import numpy

from theano import tensor

from blocks.bricks import MLP, Tanh, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from blocks.datasets import DataStream
from blocks.datasets.schemes import SequentialScheme
from blocks.main_loop import MainLoop
from blocks.algorithms import GradientDescent, SteepestDescent
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import SerializeMainLoop

from ift6266h15.code.pylearn2.datasets.variable_image_dataset import RandomCrop

from cats_vs_dogs.iterators import DogsVsCats
from cats_vs_dogs.bricks import Convolutional, Pooling, ConvMLP
from cats_vs_dogs.algorithms import Adam
from cats_vs_dogs.schemes import SequentialShuffledScheme

logging.basicConfig(level='INFO')


def parse_args():
    parser = argparse.ArgumentParser("Cats vs Dogs training algorithm")
    parser.add_argument('--image-shape',
                        default=221,
                        help='Image shape')
    parser.add_argument('--channels',
                        default=3,
                        help='Number of channels')
    parser.add_argument('--batch-size',
                        default=100,
                        help='Batch size')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    logging.info('.. starting')
    dims = [args.image_shape * args.image_shape * args.channels, 120, 2]
    mlp = MLP(activations=[Tanh(), Softmax()], dims=dims,
              weights_init=IsotropicGaussian(0.1), biases_init=Constant(0))
    mlp.initialize()

    x = tensor.matrix('X')
    y = tensor.lmatrix('y')
    y_hat = mlp.apply(x)
    cost = CategoricalCrossEntropy().apply(y, y_hat)
    error_rate = MisclassificationRate().apply(y[:, 0], y_hat)

    transformer = RandomCrop(256, args.image_shape)
    logging.info('.. model built')
    train_dataset = DogsVsCats('train', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'),
                               transformer)
    rng = numpy.random.RandomState(124)
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
        iteration_scheme=SequentialShuffledScheme(train_dataset.num_examples,
                                          args.batch_size, rng))
    valid_dataset = DogsVsCats('valid', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'),
                               transformer)
    valid_stream = DataStream(
        dataset=valid_dataset,
        iteration_scheme=SequentialShuffledScheme(train_dataset.num_examples,
                                          args.batch_size, rng))

    train_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=train_stream, prefix="train")
    valid_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=valid_stream, prefix="valid")
    test_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=test_stream, prefix="test")

    main_loop = MainLoop(
        model=mlp, data_stream=train_stream,
        algorithm=GradientDescent(
            cost=cost, step_rule=Adam()),
        extensions=[FinishAfter(after_n_epochs=50000),
                    train_monitor,
                    valid_monitor,
                    test_monitor,
                    SerializeMainLoop('./models/main.pkl'),
                    Printing()])
    main_loop.run()
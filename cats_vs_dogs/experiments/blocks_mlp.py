import logging
import os

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

logging.basicConfig()


if __name__ == '__main__':
    dims = [221 * 221 * 3, 120, 2]
    mlp = MLP(activations=[Tanh(), Softmax()], dims=dims,
              weights_init=IsotropicGaussian(0.1), biases_init=Constant(0))
    mlp.initialize()

    x = tensor.matrix('X')
    y = tensor.lmatrix('y')
    y_hat = mlp.apply(x)
    cost = CategoricalCrossEntropy().apply(y, y_hat)
    error_rate = MisclassificationRate().apply(y[:, 0], y_hat)

    transformer = RandomCrop(256, 221)
    train_dataset = DogsVsCats('train', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'),
                               transformer)
    train_stream = DataStream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples, 100))
    test_dataset = DogsVsCats('test', os.path.join('${PYLEARN2_DATA_PATH}',
                                                   'dogs_vs_cats',
                                                   'train.h5'),
                              transformer)
    test_stream = DataStream(
        dataset=test_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples, 100))
    valid_dataset = DogsVsCats('valid', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'),
                               transformer)
    valid_stream = DataStream(
        dataset=valid_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples, 100))

    train_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=train_stream, prefix="train")
    valid_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=valid_stream, prefix="valid")
    test_monitor = DataStreamMonitoring(
        variables=[cost, error_rate], data_stream=test_stream, prefix="test")

    main_loop = MainLoop(
        model=mlp, data_stream=train_stream,
        algorithm=GradientDescent(
            cost=cost, step_rule=SteepestDescent(learning_rate=1.e-4)),
        extensions=[FinishAfter(after_n_epochs=50000),
                    train_monitor,
                    valid_monitor,
                    test_monitor,
                    SerializeMainLoop('./models/main.pkl'),
                    Printing()])
    main_loop.run()
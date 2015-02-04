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

from blocks.datasets.mnist import MNIST

from cats_vs_dogs.iterators import DogsVsCats


if __name__ == '__main__':
    dims = [221 * 221 * 3, 120, 2]
    #dims = [784, 100, 10]
    mlp = MLP(activations=[Tanh(), Softmax()], dims=dims,
              weights_init=IsotropicGaussian(0.01), biases_init=Constant(0))
    mlp.initialize()

    x = tensor.matrix('X')
    y = tensor.lmatrix('y')
    y_hat = mlp.apply(x)
    cost = CategoricalCrossEntropy().apply(y, y_hat)
    error_rate = MisclassificationRate().apply(y, y_hat.T)

    train_dataset = DogsVsCats('train')
    #train_dataset = MNIST('train')
    train_stream = DataStream(
        dataset=train_dataset,
        iteration_scheme=SequentialScheme(train_dataset.num_examples, 100))
    #test_dataset = DogsVsCats('test')
    #test_stream = DataStream(
    #    dataset=test_dataset,
    #    iteration_scheme=SequentialScheme(train_dataset.num_examples, 100))
    #valid_dataset = DogsVsCats('valid')
    #valid_stream = DataStream(
    #    dataset=valid_dataset,
    #    iteration_scheme=SequentialScheme(train_dataset.num_examples, 100))
    monitor = DataStreamMonitoring(
        variables=[cost], data_stream=train_stream, prefix="test")

    main_loop = MainLoop(
        model=mlp, data_stream=train_stream,
        algorithm=GradientDescent(
            cost=cost, step_rule=SteepestDescent(learning_rate=1.e-4)),
        extensions=[FinishAfter(after_n_epochs=5),
                    monitor,
                    Printing()])
    main_loop.run()
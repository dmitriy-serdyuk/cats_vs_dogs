__author__ = 'dima'

from collections import OrderedDict
import argparse
import cPickle as pkl

import numpy as np

import theano.tensor as tt
from theano import function
import theano

from cats_vs_dogs.iterators import (SingleIterator, BatchIterator,
                                    ResizingIterator)


def main(directory, inp_size=(200, 200, 3), hid_size=40000, batch_size=200, lrate=0.01,
         epochs=50, seed=1):
    print '.. building model'
    X = tt.tensor4('X')
    y = tt.vector('y')

    rng = np.random.RandomState(seed)
    flat_inp = np.prod(inp_size)
    w_init_val = rng.normal(0, 0.01, (hid_size, flat_inp))
    W = theano.shared(w_init_val, name='W')
    b_init_val = rng.normal(0, 0.01, hid_size)
    b = theano.shared(b_init_val, name='b')
    c_init_val = rng.normal(0, 0.01, hid_size)
    c = theano.shared(c_init_val, name='c')
    params = [W, b, c]

    X_prime = X.reshape((batch_size, -1))
    h = tt.nnet.sigmoid(X_prime.dot(W.T) + b.dimshuffle('x', 0))

    out = tt.nnet.sigmoid(h.dot(c))
    cost = (-y * tt.log(out)).mean()
    grads = tt.grad(cost, params)
    updates = OrderedDict({param: param - lrate * grad for param, grad
                           in zip(params, grads)})
    make_step = function([X, y], [], updates=updates)
    compute_cost = function([X, y], cost)

    train_iter = SingleIterator(directory, 'train')
    train_iter = ResizingIterator(train_iter, inp_size[:-1])
    train_iter = BatchIterator(train_iter, batch_size)

    valid_iter = SingleIterator(directory, 'valid')
    valid_iter = ResizingIterator(valid_iter, inp_size[:-1])
    valid_iter = BatchIterator(valid_iter, batch_size)
    print '.. starting training'
    try:
        for epoch in xrange(epochs):
            train_cost = 0.
            for i, (X_val, y_val) in enumerate(train_iter):
                make_step(X_val, y_val)
                train_cost += compute_cost(X_val, y_val)
                print '.. iterations:', i, 'train cost:', train_cost / (i + 1)
            train_cost /= i
            valid_cost = 0.
            for i, (X_val, y_val) in enumerate(valid_iter):
                valid_cost += compute_cost(X_val, y_val)
            valid_cost /= i

            print '.. epoch', epoch, 'train cost', train_cost, 'valid cost', valid_cost
            print '.. saving model'
            with open('params.pkl', 'w') as fout:
                pkl.dump([param.get_value() for param in params], fout)
    except KeyboardInterrupt:
        print '.. saving model, to exit press Ctrl-C again'
        with open('params.pkl', 'w') as fout:
            pkl.dump([param.get_value() for param in params], fout)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        default='/home/dima/Downloads/datasets/cats_vs_dogs/train/',
                        help='Directory with data (extracted kaggle dataset)')
    parser.add_argument('--inp_size',
                        default=(200, 200, 3),
                        help='Input size')
    parser.add_argument('--hid_size', type=int,
                    default=1000,
                    help='Hidden layer size')
    parser.add_argument('--batch_size', type=int,
                        default=200,
                        help='Batch size')
    parser.add_argument('--lrate', type=float,
                        default=0.01,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int,
                        default=1,
                        help='Random number generator seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)


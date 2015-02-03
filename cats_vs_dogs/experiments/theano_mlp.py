__author__ = 'dima'

from collections import OrderedDict
import argparse
import cPickle as pkl
import tables

import numpy as np

import theano.tensor as tt
from theano import function
import theano

from cats_vs_dogs.iterators import (SingleIterator, BatchIterator,
                                    ResizingIterator, SingleH5Iterator)


def main(directory, inp_size=(200, 200, 3), hid_size=40000, batch_size=200,
         lrate=0.01, epochs=50, reg_coef=0.1, seed=1, model_file=''):
    print '.. building model'
    X = tt.tensor4('X')
    y = tt.vector('y')

    rng = np.random.RandomState(seed)
    flat_inp = np.prod(inp_size)

    if model_file == '':
        w_init_val = rng.normal(0, 0.01, (hid_size, flat_inp))
        W = theano.shared(w_init_val, name='W')
        b_init_val = rng.normal(0, 0.01, hid_size)
        b = theano.shared(b_init_val, name='b')
        c_init_val = rng.normal(0, 0.01, hid_size)
        c = theano.shared(c_init_val, name='c')
    else:
        param_vals = np.load(model_file)
        W_val = param_vals['W'].reshape((hid_size, flat_inp))
        b_val = param_vals['b']
        c_val = param_vals['c']
        W = theano.shared(W_val, name='W')
        b = theano.shared(b_val, name='b')
        c = theano.shared(c_val, name='c')

    params = [W, b, c]

    X_prime = X.reshape((batch_size, -1))
    h = tt.tanh(X_prime.dot(W.T) + b.dimshuffle('x', 0))

    out = tt.nnet.sigmoid(h.dot(c))

    regularizer = (W ** 2).sum() + (c ** 2).sum()
    cost = (-y * tt.log(out)).mean() + reg_coef * regularizer
    misclass = tt.neq(y, out.round()).mean()

    grads = tt.grad(cost, params)
    updates = OrderedDict({param: param - lrate * grad for param, grad
                           in zip(params, grads)})
    make_step = function([X, y], [cost, misclass], updates=updates)
    compute_costs = function([X, y], [cost, misclass])

    data_file = tables.open_file(directory)
    def get_iter(subset):
        iter = SingleH5Iterator(data_file, subset)
        iter = ResizingIterator(iter, inp_size[:-1])
        iter = BatchIterator(iter, batch_size)
        return iter

    print '.. starting training'
    try:
        for epoch in xrange(epochs):
            train_misclass = 0.
            for i, (X_val, y_val) in enumerate(get_iter('train')):
                cost, misclass = make_step(X_val, y_val)
                train_misclass += misclass
                print '.. iterations: %d train cost: %.2f misclass: %.2f' % \
                      (i, cost, misclass)
            train_misclass /= i
            valid_cost = 0.
            valid_misclass = 0.
            for i, (X_val, y_val) in enumerate(get_iter('valid')):
                cost, misclass = compute_costs(X_val, y_val)
                valid_cost += cost
                valid_misclass += misclass
            valid_cost /= i
            valid_misclass /= i

            form_string = ('.. epoch %d, train misclass %.2f, '
                           'valid cost %.2f, valid misclass %.2f' %
                           (epoch, train_misclass, valid_cost, valid_misclass))
            print form_string
            print '.. saving model'
            save_model(params)
    except KeyboardInterrupt:
        print '.. saving model, press Ctrl-C again to exit now'
        save_model(params)
        raise


def save_model(params):
    param_values = [param.get_value() for param in params]
    np.savez('params.npz', **dict(zip(['W', 'b', 'c'], param_values)))


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
    parser.add_argument('--reg_coef', type=float,
                        default=0.1,
                        help='Regularization coefficient')
    parser.add_argument('--epochs', type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--seed', type=int,
                        default=1,
                        help='Random number generator seed')
    parser.add_argument('--model_file',
                        default='',
                        help='Model to continue training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(**args.__dict__)


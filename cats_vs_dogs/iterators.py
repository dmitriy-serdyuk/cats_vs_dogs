__author__ = 'dima'

import os
import cPickle as pkl
from itertools import izip

import numpy as np
from scipy import misc

from blocks.datasets import Dataset
from ift6266h15.code.pylearn2.datasets import variable_image_dataset
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import RandomCrop


class SingleIterator(object):
    def __init__(self, directory, subset, floatX='float32'):
        load_path = os.path.join(directory, '../datasets.pkl')
        with open(load_path, 'r') as fin:
            train, valid, test = pkl.load(fin)
        if subset == 'train':
            self.data_files = train
        elif subset == 'valid':
            self.data_files = valid
        elif subset == 'test':
            self.data_files = test
        else:
            raise ValueError('Incorrect subset, possible values are train, '
                             'valid, or test')
        self.directory = directory
        self.floatX = floatX

    def __iter__(self):
        for data_file in self.data_files:
            full_path = os.path.join(self.directory, data_file)
            with open(full_path, 'r') as fin:
                image, label = pkl.load(fin)
            yield image, np.array(label, dtype=self.floatX)


class SingleH5Iterator(object):
    def __init__(self, data_file, subset, floatX='float32'):
        self.X = data_file.get_node('/' + subset + '/X')
        self.y = data_file.get_node('/' + subset + '/y')
        self.s = data_file.get_node('/' + subset + '/s')
        self.floatX = floatX

    def __iter__(self):
        for X, y, shape in izip(self.X.iterrows(), self.y.iterrows(),
                               self.s.iterrows()):
            yield (np.cast[self.floatX](X.reshape(shape)),
                   np.cast[self.floatX](y))


class ResizingIterator(object):
    def __init__(self, iterator, size):
        self.iterator = iterator.__iter__()
        self.size = size

    def __iter__(self):
        return self

    def next(self):
        image, label = self.iterator.next()
        return misc.imresize(image, self.size), label


class BatchIterator(object):
    def __init__(self, iterator, batch_size):
        self.iterator = iterator.__iter__()
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        images = []
        labels = []
        for i in xrange(self.batch_size):
            image, label = self.iterator.next()
            images += [image]
            labels += [label]
        return np.array(images), np.array(labels)


class DogsVsCats(Dataset):
    provides_sources = ['X', 'y']

    def __init__(self, subset):
        self.sources = ['X', 'y']
        self.subset = subset
        if subset == 'train':
            self.start = 0
            self.stop = 200
        elif subset == 'valid':
            self.start = 20000
            self.stop = 22500
        elif subset == 'test':
            self.start = 22500
            self.stop = 25000
        container = variable_image_dataset.DogsVsCats(RandomCrop(256, 221),
                                                      start=self.start,
                                                      stop=self.stop)
        self.iterator = container.iterator(
            mode='batchwise_shuffled_sequential',
            batch_size=100)
        super(DogsVsCats, self).__init__(self.sources)

    def num_examples(self):
        return self.stop - self.start

    def get_data(self, state=None, request=None):
        X, y = next(self.iterator)
        X = X.transpose(0, 3, 1, 2).reshape((100, -1))
        y = np.concatenate((y, 1 - y), axis=1)
        return X, y


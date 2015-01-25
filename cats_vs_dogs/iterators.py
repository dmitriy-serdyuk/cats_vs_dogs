__author__ = 'dima'

import os
import cPickle as pkl

import numpy as np
from scipy import misc


class SingleIterator(object):
    def __init__(self, directory, subset):
        load_path = os.path.join(directory, '../datasets.pkl')
        with open(load_path, 'r') as fin:
            train, valid, test = pkl.load(fin)
        if subset == 'train':
            self.data_files = train
        elif subset == 'valid':
            self.data_files = valid
        elif subset == 'train':
            self.data_files = test
        else:
            raise ValueError('Incorrect subset, possible values are train, '
                             'valid, or test')
        self.directory = directory

    def __iter__(self):
        return self

    def next(self):
        for data_file in self.data_files:
            full_path = os.path.join(self.directory, data_file)
            with open(full_path, 'r') as fin:
                image, label = pkl.load(fin)
            yield image, label
        raise StopIteration()


class ResizingIterator(object):
    def __init__(self, iterator, size):
        self.iterator = iterator
        self.size = size

    def __iter__(self):
        return self

    def next(self):
        image, label = self.iterator.next()
        return misc.imresize(image, self.size), label


class BatchIterator(object):
    def __init__(self, iterator, batch_size):
        self.iterator = iterator
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

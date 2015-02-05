__author__ = 'dima'

import os
import cPickle as pkl
from itertools import izip
import tables

import numpy as np
from scipy import misc

import theano

from blocks.datasets import Dataset
from ift6266h15.code.pylearn2.datasets import variable_image_dataset
from ift6266h15.code.pylearn2.datasets.variable_image_dataset import RandomCrop

from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets import cache


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

    def __init__(self, subset, path, transformer):
        self.sources = ['X', 'y']
        self.subset = subset
        self.path = path
        self.data_node = 'Data'
        self.rescale = 256
        self.transformer = transformer
        self.floatX = theano.config.floatX
        if subset == 'train':
            self.start = 0
            self.stop = 200
        elif subset == 'valid':
            self.start = 20000
            self.stop = 22500
        elif subset == 'test':
            self.start = 22500
            self.stop = 25000
        super(DogsVsCats, self).__init__(self.sources)

    def open(self):
        # Locally cache the files before reading them
        path = preprocess(self.path)
        datasetCache = cache.datasetCache
        path = datasetCache.cache_file(path)

        h5file = tables.openFile(path, mode="r")
        node = h5file.getNode('/', self.data_node)

        self.rescale = float(self.rescale)
        X = getattr(node, 'X')
        s = getattr(node, 's')
        y = getattr(node, 'y')
        print 'data opened'

        return h5file, X, y, s

    def close(self, state):
        h5file, _, _, _ = state
        h5file.close()

    def num_examples(self):
        return self.stop - self.start

    def get_data(self, state=None, request=None):
        indexes = slice(request[0] + self.start, request[-1] + 1 + self.start)
        _, X, y, s = state
        images = X[indexes]
        targets = y[indexes]
        shapes = s[indexes]
        out_shape = self.transformer.get_shape()
        shape_x, shape_y = out_shape
        n_channels = images[0].shape[0] / shapes[0][0] / shapes[0][1]
        X_buffer = np.zeros((len(request), shape_x, shape_y, n_channels),
                            dtype=self.floatX)
        for i, (img, s) in enumerate(izip(images, shapes)):
            # Transpose image in 'b01c' format to comply with
            # transformer interface
            b01c = img.reshape(s)
            # Assign i'th example in the batch with the preprocessed
            # image
            X_buffer[i] = self.transformer(b01c)
        X = X_buffer.transpose(0, 3, 1, 2).reshape((len(request), -1))
        y = np.concatenate((targets, 1 - targets), axis=1)
        return X, y


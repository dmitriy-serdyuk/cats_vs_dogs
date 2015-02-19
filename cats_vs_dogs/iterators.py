__author__ = 'dima'

import os
import cPickle as pkl
from itertools import izip
import tables
from picklable_itertools import izip

import numpy as np
from scipy import misc

import theano

from blocks.datasets import Dataset
from blocks.datasets.streams import DataStreamWrapper

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
    provides_sources = ['X', 'y', 'shape']

    def __init__(self, subset, path, transformer, flatten=False):
        self.subset = subset
        self.path = path
        self.data_node = 'Data'
        self.rescale = 256
        self.transformer = transformer
        self.floatX = theano.config.floatX
        self.n_channels = 3
        self.flatten = flatten
        if subset == 'train':
            self.start = 0
            self.stop = 20000
        elif subset == 'valid':
            self.start = 20000
            self.stop = 22500
        elif subset == 'test':
            self.start = 22500
            self.stop = 25000
        self.num_examples = self.stop - self.start
        # Locally cache the files before reading them
        path = preprocess(self.path)
        datasetCache = cache.datasetCache
        self.path = datasetCache.cache_file(path)
        h5file = tables.openFile(self.path, mode="r")
        node = h5file.getNode('/', self.data_node)

        self.rescale = float(self.rescale)
        self.X = getattr(node, 'X')
        self.s = getattr(node, 's')
        self.y = getattr(node, 'y')
        super(DogsVsCats, self).__init__(self.provides_sources)

    def get_data(self, state=None, request=None):
        if not request:
            raise StopIteration
        indexes = slice(request[0] + self.start, request[-1] + 1 + self.start)
        if indexes.stop > self.stop:
            raise StopIteration
        X, y, s = self.X, self.y, self.s
        images = X[indexes]
        targets = y[indexes]
        shapes = s[indexes]
        if self.flatten:
            X = images.transpose(0, 3, 1, 2).reshape((len(request), -1))
        else:
            X = images.transpose(0, 3, 1, 2)
        y = np.concatenate((targets, 1 - targets), axis=1)
        return X / 256. - .5, y, shapes


class ReshapeStream(DataStreamWrapper):
    def __init__(self, **kwargs):
        super(ReshapeStream, self).__init__(**kwargs)

    @property
    def sources(self):
        return [source for source in self.data_stream.sources
                if source != 'shape']

    def get_data(self, request=None):
        X, y, s = next(self.child_epoch_iterator)
        X = X.reshape(s)
        return X, y


class UnbatchStream(DataStreamWrapper):
    def __init__(self, **kwargs):
        self.data = None
        super(UnbatchStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        if not self.data:
            X_batch, y_batch = next(self.child_epoch_iterator)
            self.data = izip(X_batch, y_batch)

        try:
            return self.data
        except StopIteration:
            self.data = None
            return self.get_data()


class RandomCropStream(DataStreamWrapper):
    """
    Crops a square at random on a rescaled version of the image

    Parameters
    ----------
    scaled_size : int
        Size of the smallest side of the image after rescaling
    crop_size : int
        Size of the square crop. Must be bigger than scaled_size.
    rng : int or rng, optional
        RNG or seed for an RNG
    """
    _default_seed = 2015 + 1 + 18

    def __init__(self, scaled_size, crop_size, rng, **kwargs):
        super(RandomCropStream, self).__init__(**kwargs)
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        if not self.scaled_size > self.crop_size:
            raise ValueError('Scaled size should be greater than crop size')
        if rng:
            self.rng = rng
        else:
            self.rng = np.RandomState(self._default_seed)

    def get_shape(self):
        return (self.crop_size, self.crop_size)

    def get_data(self, request=None):
        X, y = next(self.child_epoch_iterator)
        small_axis = np.argmin(X.shape[:-1])
        ratio = (1.0 * self.scaled_size) / X.shape[small_axis]
        resized_image = misc.imresize(X, ratio)

        max_i = resized_image.shape[0] - self.crop_size
        max_j = resized_image.shape[1] - self.crop_size
        i = self.rng.randint(low=0, high=max_i)
        j = self.rng.randint(low=0, high=max_j)
        cropped_image = resized_image[i: i + self.crop_size,
                        j: j + self.crop_size, :]
        return cropped_image, y


__author__ = 'dima'

import os
import cPickle as pkl
import math
import tables
from picklable_itertools import izip

import numpy as np
from scipy import misc

import theano

from blocks.datasets import Dataset
from blocks.datasets.streams import DataStreamWrapper

from pylearn2.utils.string_utils import preprocess
from pylearn2.datasets import cache

floatX = theano.config.floatX


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


class Hdf5Dataset(Dataset):
    def __init__(self, sources, start, stop, path, data_node='Data',
                 sources_in_file=None):
        if sources_in_file is None:
            sources_in_file = sources
        self.sources_in_file = sources_in_file
        self.provides_sources = sources
        self.path = path
        self.data_node = data_node
        self.start = start
        self.stop = stop
        self.num_examples = self.stop - self.start
        # Locally cache the files before reading them
        path = preprocess(self.path)
        datasetCache = cache.datasetCache
        self.path = datasetCache.cache_file(path)
        h5file = tables.openFile(self.path, mode="r")
        node = h5file.getNode('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]
        super(Hdf5Dataset, self).__init__(self.provides_sources)

    def get_data(self, state=None, request=None):
        if not request:
            raise StopIteration
        data = [node[request] for node in self.nodes]
        return data

    def __getstate__(self):
        dict = self.__dict__
        return {key: val for key, val in dict.iteritems() if key != 'nodes'}

    def __setstate__(self, state):
        self.__dict__ = state
        h5file = tables.openFile(self.path, mode="r")
        node = h5file.getNode('/', self.data_node)

        self.nodes = [getattr(node, source) for source in self.sources_in_file]


class DogsVsCats(Hdf5Dataset):
    provides_sources = ['X', 'y', 'shape']

    def __init__(self, subset, path):
        if subset == 'train':
            start = 0
            stop = 20000
        elif subset == 'valid':
            start = 20000
            stop = 22500
        elif subset == 'test':
            start = 22500
            stop = 25000
        else:
            raise ValueError('Subset should be train, valid, or test')

        super(DogsVsCats, self).__init__(self.provides_sources, start, stop,
                                         path, sources_in_file=['X', 'y', 's'])


class OneHotEncoderStream(DataStreamWrapper):
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(OneHotEncoderStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        X, y, s = next(self.child_epoch_iterator)
        batch_size = y.shape[0]
        out_y = np.zeros((batch_size, self.num_classes), dtype='int64')
        out_y[(xrange(batch_size), y[:, 0])] = 1
        return X, out_y, s


class ReshapeStream(DataStreamWrapper):
    def __init__(self,  **kwargs):
        super(ReshapeStream, self).__init__(**kwargs)

    @property
    def sources(self):
        return [source for source in self.data_stream.sources
                if source != 'shape']

    def get_data(self, request=None):
        X, y, s = next(self.child_epoch_iterator)
        X = X.reshape(s)
        return X, y


class ImageTransposeStream(DataStreamWrapper):
    def get_data(self, request=None):
        X, y = next(self.child_epoch_iterator)
        return X.transpose(0, 3, 1, 2), y


class UnbatchStream(DataStreamWrapper):
    def __init__(self, **kwargs):
        self.data = None
        super(UnbatchStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        if not self.data:
            data = next(self.child_epoch_iterator)
            self.data = izip(*data)

        try:
            return next(self.data)
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
        return np.cast[floatX](cropped_image) / 256. - .5, y


class RandomRotateStream(DataStreamWrapper):
    """Rotates image.

    Rotates image on a random angle in order to get another one
    of desired size. The maximum rotation angle is computed to be
    able to inscribe the output image into the rotated input image.

    Parameters
    ----------
    input_size : int
        The size of a side of a square image.
    output_size : int
        Desired output size.
    rng : :class:`~random.RandomState`
        Random number generator

    """
    def __init__(self, input_size, output_size, rng, **kwargs):
        self.max_angle = math.asin((input_size ** 2 - output_size ** 2) /
                                   float(input_size * output_size))
        self.rng = rng
        self.input_size = input_size
        self.output_size = output_size
        super(RandomRotateStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        X, y = next(self.child_epoch_iterator)
        sample_angle = (self.ng.random_sample() - 0.5) * 2 * self.max_angle
        new_image = misc.ndimage.interpolation.rotate(X, sample_angle)
        start = (self.input_size - self.output_size) / 2.
        stop = self.output_size - (self.input_size - self.output_size) / 2.
        reshaped = new_image[start, stop, :]
        return reshaped, y

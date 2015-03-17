import os
import cPickle as pkl
from collections import OrderedDict, deque
import math
import tables
from picklable_itertools import izip, chain

import numpy as np
import scipy
from scipy import misc, ndimage

import theano

from fuel.datasets import Dataset
from fuel.transformers import Transformer

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


class OneHotEncoderStream(Transformer):
    def __init__(self, num_classes, target_source, **kwargs):
        self.num_classes = num_classes
        self.target_source = target_source
        super(OneHotEncoderStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        target = data[self.target_source]

        batch_size = target.shape[0]
        out_target = np.zeros((batch_size, self.num_classes), dtype='int64')
        out_target[(xrange(batch_size), target[0])] = 1

        data[self.target_source] = out_target
        return list(data.values())


class ReshapeStream(Transformer):
    def __init__(self,  image_source, shape_source, **kwargs):
        self.image_source = image_source
        self.shape_source = shape_source
        super(ReshapeStream, self).__init__(**kwargs)

    @property
    def sources(self):
        return [source for source in self.data_stream.sources
                if source != self.shape_source]

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        image = data[self.image_source]
        shape = data[self.shape_source]

        image = np.array(image).reshape(shape)

        data[self.image_source] = image
        data.pop(self.shape_source)
        return list(data.values())


class ImageTransposeStream(Transformer):
    def __init__(self, image_source, **kwargs):
        self.image_source = image_source
        super(ImageTransposeStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))

        image = data[self.image_source]
        image = image.transpose(0, 3, 1, 2)
        data[self.image_source] = image
        return list(data.values())


class UnbatchStream(Transformer):
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


class RandomCropStream(Transformer):
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

    def __init__(self, scaled_size, crop_size, image_source, rng, **kwargs):
        self.scaled_size = scaled_size
        self.crop_size = crop_size
        self.image_source = image_source
        if not self.scaled_size > self.crop_size:
            raise ValueError('Scaled size should be greater than crop size')
        if rng:
            self.rng = rng
        else:
            self.rng = np.RandomState(self._default_seed)
        super(RandomCropStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        image = data[self.image_source]

        small_axis = np.argmin(image.shape[:-1])
        ratio = (1.0 * self.scaled_size) / image.shape[small_axis]
        resized_image = misc.imresize(image, ratio)

        max_i = resized_image.shape[0] - self.crop_size
        max_j = resized_image.shape[1] - self.crop_size
        i = self.rng.randint(low=0, high=max_i)
        j = self.rng.randint(low=0, high=max_j)
        cropped_image = resized_image[i: i + self.crop_size,
                                      j: j + self.crop_size, :]
        image = np.cast[floatX](cropped_image) / 256. - .5
        data[self.image_source] = image
        return list(data.values())


class RandomRotateStream(Transformer):
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
    def __init__(self, input_size, output_size, image_source, rng, **kwargs):
        self.max_angle = math.asin((input_size ** 2 - output_size ** 2) /
                                   float(input_size * output_size))
        self.rng = rng
        self.input_size = input_size
        self.output_size = output_size
        self.image_source = image_source
        super(RandomRotateStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))
        image = data[self.image_source]

        sample_angle = (self.rng.random_sample() - 0.5) * 2 * self.max_angle
        new_image = ndimage.interpolation.rotate(image, sample_angle)
        start = (self.input_size - self.output_size) / 2.
        stop = self.output_size - (self.input_size - self.output_size) / 2.
        reshaped = new_image[start: stop, start: stop, :]

        data[self.image_source] = reshaped
        return list(data.values())


class SourceSelectStream(Transformer):
    def __init__(self, pool, source, **kwargs):
        self.source = source
        self.pool = pool
        self.provides_sources = [source]
        self.sources = [source]
        super(SourceSelectStream, self).__init__(**kwargs)

    def get_data(self, request=None):
        return self.pool.get_source(self.source)


class SelectStreamPool(object):
    def __init__(self, data_stream):
        self.pool = OrderedDict()
        self.data_stream = data_stream
        self.sources = data_stream.sources
        for source in data_stream.sources:
            self.pool[source] = deque()

    def get_streams(self):
        streams = [SourceSelectStream(self, source,
                                      data_stream=self.data_stream)
                   for source in self.sources]
        return streams

    def get_source(self, source):
        if not self.pool[source]:
            source_vals = self.data_stream.get_data()
            for name, val in zip(self.sources, source_vals):
                self.pool[name].appendleft(val)
        return self.pool[source].pop(),


class MergeStream(Transformer):
    def __init__(self, streams, **kwargs):
        self.streams = streams
        self.sources = list(chain(*[stream.sources for stream in streams]))
        super(MergeStream, self).__init__(data_stream=streams[0], **kwargs)

    def get_data(self, request=None):
        next_data = []
        for iterator in self.stream_epoch_iterators:
            val = next(iterator)
            next_data.extend(val)
        return next_data

    def get_epoch_iterator(self, **kwargs):
        self.stream_epoch_iterators = [stream.get_epoch_iterator()
                                       for stream in self.streams]
        return super(MergeStream, self).get_epoch_iterator(**kwargs)

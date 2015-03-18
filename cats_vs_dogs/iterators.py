import os
import cPickle as pkl
import math
from abc import abstractmethod, ABCMeta
from collections import OrderedDict
from picklable_itertools import izip

import numpy as np
from scipy import misc, ndimage

import theano

from fuel.datasets.hdf5 import Hdf5Dataset
from fuel.transformers import Transformer

floatX = theano.config.floatX


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


class DataTransformer(Transformer):
    __metaclass__ = ABCMeta

    @abstractmethod
    def transform_data(self, data):
        pass

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        data = OrderedDict(zip(self.data_stream.sources, data))

        out_data = self.transform_data(data)

        return [value for name, value in out_data.iteritems()
                if name in self.sources]


class OneHotEncoderStream(DataTransformer):
    def __init__(self, num_classes, target_source, **kwargs):
        self.num_classes = num_classes
        self.target_source = target_source
        super(OneHotEncoderStream, self).__init__(**kwargs)

    def transform_data(self, data):
        target = data[self.target_source]

        batch_size = target.shape[0]
        out_target = np.zeros((batch_size, self.num_classes), dtype='int64')
        out_target[(xrange(batch_size), target[0])] = 1
        data[self.target_source] = out_target
        return data


class Reshape(DataTransformer):
    def __init__(self,  image_source, shape_source, **kwargs):
        self.image_source = image_source
        self.shape_source = shape_source
        super(Reshape, self).__init__(**kwargs)

    @property
    def sources(self):
        return [source for source in self.data_stream.sources
                if source != self.shape_source]

    def transform_data(self, data):
        image = data[self.image_source]
        shape = data[self.shape_source]

        image = np.array(image).reshape(shape)

        data[self.image_source] = image
        return data


class ImageTranspose(DataTransformer):
    def __init__(self, image_source, **kwargs):
        self.image_source = image_source
        super(ImageTranspose, self).__init__(**kwargs)

    def transform_data(self, data):
        image = data[self.image_source]
        image = image.transpose(0, 3, 1, 2)
        data[self.image_source] = image
        return data


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


class RandomCrop(DataTransformer):
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
        super(RandomCrop, self).__init__(**kwargs)

    def transform_data(self, data):
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
        data[self.image_source] = cropped_image
        return data


class Normalize(DataTransformer):
    def __init__(self, image_source, **kwargs):
        self.image_source = image_source
        super(Normalize, self).__init__(**kwargs)

    def transform_data(self, data):
        image = data[self.image_source]

        image = np.cast[floatX](image) / 256. - .5
        data[self.image_source] = image
        return data


class RandomRotate(DataTransformer):
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
        super(RandomRotate, self).__init__(**kwargs)

    def transform_data(self, data):
        image = data[self.image_source]

        sample_angle = (self.rng.random_sample() - 0.5) * self.max_angle
        new_image = ndimage.interpolation.rotate(image, sample_angle /
                                                 math.pi * 180.)
        new_image_size = new_image.shape[0]
        start = (new_image_size - self.output_size) / 2.
        stop = new_image_size - start
        reshaped = new_image[start: stop, start: stop, :]

        data[self.image_source] = reshaped
        return data

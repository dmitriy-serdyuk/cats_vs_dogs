__author__ = 'serdyuk'


import numpy
import theano
import logging
import os

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.dump import MainLoopDumpManager
from blocks.utils import reraise_as


import logging
import signal
import time
from subprocess import Popen, PIPE

import numpy
try:
    from bokeh.plotting import (curdoc, cursession, figure, output_server,
                                push, show)
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
from blocks.extensions import SimpleExtension
logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"


class DumpWeights(SimpleExtension):
    """Dumps the weights only.

    Makes a `SAVED_TO` record in the log with the dumping destination
    in the case of success and ``None`` in the case of failure.

    Parameters
    ----------
    state_path : str
        The folder to dump the weights to. Will be created it does not
        exist.

    Notes
    -----
    Requires the model to be a Brick or a list of Bricks.

    """
    def __init__(self, state_path, **kwargs):
        kwargs.setdefault("after_training", True)
        super(DumpWeights, self).__init__(**kwargs)
        self.manager = MainLoopDumpManager(state_path)

    def do(self, callback_name, **kwargs):
        try:
            self.main_loop.log.current_row[SAVED_TO] = (
                self.manager.folder)
            self.manager.dump_parameters(self.main_loop)
        except:
            self.main_loop.log.current_row[SAVED_TO] = None
            raise


class LoadWeights(TrainingExtension):
    """Loads a dump of weights into the main loop.

    Makes a `LOADED_FROM` record in the log with the dump path.

    Parameters
    ----------
    state_path : str
        The path to the folder with dump.

    Notes
    -----
    Requires the model to be a Brick or a list of Bricks.

    """
    def __init__(self, state_path, **kwargs):
        super(LoadWeights, self).__init__(**kwargs)
        self.manager = MainLoopDumpManager(state_path)

    def before_training(self):
        if not os.path.exists(self.manager.folder):
            logger.info("No dump found")
            return
        logger.info("Loading the state from {} into the main loop"
                    .format(self.manager.folder))
        try:
            parameters= self.manager.load_parameters()
            inject_parameter_values(self.main_loop.model, parameters)
            self.main_loop.log.current_row[LOADED_FROM] = self.manager.folder
        except:
            reraise_as("Failed to load the state")


def get_grid_shape(num_examples):
    """Get the best grid shape given a number of examples.

    Parameters
    ----------
    num_examples : int
        Number of examples considered.


    Returns
    -------
    (height, width) : tuple
        The grid shape.

    """
    height = int(numpy.floor(numpy.sqrt(num_examples)))
    width = int(numpy.ceil(num_examples * 1. / height))

    return (height, width)


def gather_patches(raw_data, grid_shape=None):
    """Gather patches in one image.

    Parameters
    ----------
    raw_data : numpy.array
        The image 4D tensor with axes ('b', 0, 1, 'c').

    Returns
    -------
    image : numpy.array
        The image 3D tensor with axes (0, 1, 'c')

    """
    num_examples = raw_data.shape[0]
    img_h = raw_data.shape[1]
    img_w = raw_data.shape[2]
    img_c = raw_data.shape[3]

    if grid_shape is None:
        grid_shape = get_grid_shape(num_examples)

    expected_examples = grid_shape[0] * grid_shape[1]
    padding_pattern = (((0, expected_examples
                         - num_examples),)
                       + ((0, 0),) * 3)
    padded_data = numpy.pad(
        raw_data,
        pad_width=padding_pattern,
        mode='constant',
        constant_values=0
    )

    image = padded_data.view().reshape((
        grid_shape[1], grid_shape[0] * img_h, img_w, img_c)
    ).transpose(
        (1, 0, 2, 3)
    ).reshape(
        (grid_shape[0] * img_h,
         grid_shape[1] * img_w,
         img_c)
    ).copy()

    image *= 0.5
    image += 0.5
    image *= 255

    return image


def reshape_image(raw_data, image_shape,
                  orig_axes=(0, 1, 'c'), dest_axes=(0, 1, 'c')):
    """Reshape an data tensor into a 4D image tensor.

    Parameters
    ----------
    raw_data : numpy.array
        The data tensor.
    image_shape : tuple
        The original image shape. To match with the original axes.
        See orig_axes.
    orig_axes : tuple, optional
        The original axes of the image. Default is (0, 1, 'c').
    dest_axes : tuple, optional
        The desired axes. Default is (0, 1, 'c').

    Returns
    -------
    image : numpy.array
        The image 4D tensor with axes corresponding to dest_axes.

    """

    if raw_data.ndim != len(image_shape) + 1:
        image_tensor = raw_data.reshape((-1,) + image_shape)
    else:
        image_tensor = raw_data

    rval = image_tensor.transpose(
        (0,) + tuple(orig_axes.index(x) + 1 for x in dest_axes)
    )

    return rval[::-1, ::-1]


class ImageGetter(object):
    """A base class for getting image tensor to display.

    Parameters
    ----------
    image_shape : tuple of int
        The shape of one image.
    axes : tuple, optional
        The original axes of the image.
    shift : float, optional
        The shift necessary so that the images are between -1 and 1.
        Default is 0.
    rescale : float, optional
        The rescale necessary so that the images are between -1 and 1.
        The output images follow the formula:
        images = rescale * (init_images + shift)
        Default is 1.
    grid_shape : tuple, optional
        The shape of the grid that will be shown. Default is inferred.

    """
    default_axes = (0, 1, 'c')

    def __init__(self, image_shape, axes=(0, 1, 'c'),
                 shift=0, rescale=1, grid_shape=None):
        self.image_shape = image_shape
        self.axes = axes
        self.shift = shift
        self.rescale = rescale
        self.grid_shape = grid_shape

    def get_image_data(self):
        """Get raw image data.

        The data will later be reshaped, transposed and rescaled to be
        between -1 and 1.

        """
        raise NotImplementedError(str(type(self)) + 'does not'
                                  'implement get_image.')

    def __call__(self):
        raw_data = self.get_image_data()

        image_tensor = reshape_image(
            raw_data=raw_data,
            image_shape=self.image_shape,
            orig_axes=self.axes,
            dest_axes=self.default_axes
        )

        image = gather_patches(
            numpy.clip(
                self.rescale * (image_tensor + self.shift),
                -1, 1
            ),
            self.grid_shape
        )

        return image


class ImageDataStreamDisplay(ImageGetter):
    """Return image of image examples from a data stream.

    Parameters
    ----------
    data_stream : fuel.streams.DataStream
        The indexable image dataset you show image from.
    num_examples : int, optional
        The number of image examples shown. Default is 100.
    source : string, optional
        The source from which the image is obtained. Default is 'features'.

    """

    def __init__(self, data_stream, num_examples=100,
                 source='features', ** kwargs):
        self.data_stream = data_stream
        self.num_examples = num_examples
        self.source = source
        self.iterator = data_stream.get_epoch_iterator()

        super(ImageDataStreamDisplay, self).__init__(** kwargs)

    def get_image_data(self):
        try:
            raw_data = next(
                self.iterator
            )[self.data_stream.sources.index(self.source)][:self.num_examples]
        except StopIteration:
            self.iterator = self.data_stream.get_epoch_iterator()
            raw_data = next(
                self.iterator
            )[self.data_stream.sources.index(self.source)][:self.num_examples]

        return raw_data


class WeightDisplay(ImageGetter):
    """Return image from weights parameters.

    Parameters
    ----------
    weights : theano shared variable
        The Theano shared variable that represent the weights.
    n_weights : int, optional

    """

    def __init__(self, weights, n_weights=None, transpose=None, ** kwargs):
        self.weights = weights
        self.n_weights = n_weights
        self.transpose = transpose

        super(WeightDisplay, self).__init__(** kwargs)

    def get_image_data(self):
        raw_data = self.weights.get_value()

        if self.transpose is not None:
            raw_data = raw_data.transpose(self.transpose)

        if self.n_weights is not None:
            raw_data = raw_data[:self.n_weights]

        # =raw_data -= raw_data.min()
        raw_data /= numpy.maximum(
            abs(raw_data).max(axis=tuple(range(1, raw_data.ndim)),
                keepdims=True),
            1e-8)
        # raw_data /= max(abs(raw_data.max()), 1e-8)
        raw_data *= .5
        raw_data += .5

        return raw_data


class ImageSamplesDisplay(ImageGetter):
    def __init__(self, samples, ** kwargs):
        self.sample = theano.function([], samples)

        super(ImageSamplesDisplay, self).__init__(** kwargs)

    def get_image_data(self):
        raw_data = self.sample()

        return raw_data


class PlotManager(SimpleExtension):
    """Live plotting.

    In most cases it is preferable to start the Bokeh plotting server
    manually, so that your plots are stored permanently.

    Alternatively, you can set the ``start_server`` argument of this
    extension to ``True``, to automatically start a server when training
    starts. However, in that case your plots will be deleted when you shut
    down the plotting server!

    .. warning::

       When starting the server automatically using the ``start_server``
       argument, the extension won't attempt to shut down the server at the
       end of training (to make sure that you do not lose your plots the
       moment training completes). You have to shut it down manually (the
       PID will be shown in the logs). If you don't do this, this extension
       will crash when you try and train another model with
       ``start_server`` set to ``True``, because it can't run two servers
       at the same time.

    Parameters
    ----------
    document : str
        The name of the Bokeh document. Use a different name for each
        experiment if you are storing your plots.
    channels : list of lists of strings
        The names of the monitor channels that you want to plot. The
        channels in a single sublist will be plotted together in a single
        figure, so use e.g. ``[['test_cost', 'train_cost'],
        ['weight_norms']]`` to plot a single figure with the training and
        test cost, and a second figure for the weight norms.
    open_browser : bool, optional
        Whether to try and open the plotting server in a browser window.
        Defaults to ``True``. Should probably be set to ``False`` when
        running experiments non-locally (e.g. on a cluster or through SSH).
    start_server : bool, optional
        Whether to try and start the Bokeh plotting server. Defaults to
        ``False``. The server started is not persistent i.e. after shutting
        it down you will lose your plots. If you want to store your plots,
        start the server manually using the ``bokeh-server`` command. Also
        see the warning above.

    """
    def __init__(self, document, plotters, open_browser=False,
                 start_server=False, **kwargs):
        if not BOKEH_AVAILABLE:
            raise ImportError
        self.plotters = plotters
        self.start_server = start_server
        self.document = document
        self._startserver()

        for plotter in self.plotters:
            plotter.manager = self
            plotter.initialize()
        if open_browser:
            show()

        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault("before_first_epoch", True)
        super(PlotManager, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        for plotter in self.plotters:
            plotter.call()

        push()

    def _startserver(self):
        if self.start_server:
            def preexec_fn():
                """Prevents the server from dying on training interrupt."""
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Only memory works with subprocess, need to wait for it to start
            logger.info('Starting plotting server on localhost:5006')
            self.sub = Popen('bokeh-server --ip 0.0.0.0 '
                             '--backend memory'.split(),
                             stdout=PIPE, stderr=PIPE, preexec_fn=preexec_fn)
            time.sleep(2)
            logger.info('Plotting server PID: {}'.format(self.sub.pid))
        else:
            self.sub = None
        output_server(self.document)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['sub'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._startserver()
        curdoc().add(*self.p)


class BasePlotter(object):
    """Base class for plotter."""
    @property
    def manager(self):
        return self._manager

    @manager.setter
    def manager(self, manager):
        self._manager = manager

    def initialize(self):
        """Initialize the plots state."""
        pass

    def call(self):
        """Update the plots"""
        pass

    def set_titles(self, titles, matching_list):
        if isinstance(titles, str):
            self.titles = [titles] * len(matching_list)
        else:
            if not isinstance(titles, list):
                raise ValueError('titles argument must be a '
                                 'list of strings or a string.')
            if len(titles) == 1:
                self.titles = titles * len(matching_list)
            elif len(titles) != len(matching_list):
                raise ValueError('titles must have same size as '
                                 'channels.')
            else:
                self.titles = titles


class Plotter(BasePlotter):
    """Plotting of monitoring channels.

    channels : list of lists of strings
        The names of the monitor channels that you want to plot. The
        channels in a single sublist will be plotted together in a single
        figure, so use e.g. ``[['test_cost', 'train_cost'],
        ['weight_norms']]`` to plot a single figure with the training and
        test cost, and a second figure for the weight norms.
    titles : list of strings or string
        The name of the associated plots. If t

    """
    # Tableau 10 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, channels, titles=''):
        self.channels = channels
        self.set_titles(titles, channels)

        super(Plotter, self).__init__()

    def initialize(self):
        channels = self.channels
        titles = self.titles
        self.plots = {}
        self.p = []
        self.p_indices = {}
        self.color_mapping = {}
        j = 0
        for i, (channel_set, title) in enumerate(zip(channels, titles)):
            self.p.append(figure(title=title))
            for channel in channel_set:
                self.p_indices[channel] = i
                if channel not in self.color_mapping.keys():
                    self.color_mapping[channel] = self.colors[j]
                    j += 1

    def call(self):
        log = self.manager.main_loop.log
        iteration = log.status['iterations_done']
        i = 0
        for key, value in log.current_row.items():
            if key in self.p_indices:
                if key not in self.plots:
                    fig = self.p[self.p_indices[key]]
                    fig.line([iteration], [value], legend=key,
                             x_axis_label='iterations',
                             y_axis_label='value', name=key,
                             line_color=self.color_mapping[key])
                    i += 1
                    renderer = fig.select(dict(name=key))
                    self.plots[key] = renderer[0].data_source
                else:
                    self.plots[key].data['x'].append(iteration)
                    self.plots[key].data['y'].append(value)

                    cursession().store_objects(self.plots[key])


class DisplayImage(BasePlotter):
    """Show images.

    Parameters
    ----------
    image_getters : list of callable objects
        A list of function without argument returning images to display.
        The returned images must be in (N, M, 3), (N, M, 1) or (N, M) and
        must be between 0 and 255.
    """

    def __init__(self, image_getters, titles=''):
        self.image_getters = image_getters
        self.set_titles(titles, image_getters)

        super(DisplayImage, self).__init__()

    def initialize(self):
        self.figs = []
        for image_getter, title in zip(self.image_getters, self.titles):
            self.figs.append(figure(title=title,
                                    x_range=[0, 1],
                                    y_range=[0, 1]))

    def call(self):
        for fig, image_getter in zip(self.figs, self.image_getters):
            rgba_image = format_image_to_rgba(image_getter()).copy()
            img_h = rgba_image.shape[0]
            img_w = rgba_image.shape[1]
            fig.renderers = []

            fig.image_rgba(image=[rgba_image],
                           x=[0], y=[0],
                           dw=[min(img_w * 1. / img_h, 1)],
                           dh=[min(img_h * 1. / img_w, 1)])


def format_image_to_rgba(image_0255):
    image_0255_clipped = numpy.clip(image_0255, 0, 255)
    int_image = numpy.round(
        image_0255_clipped
    ).astype(dtype=numpy.uint8)

    rgba_image = numpy.zeros(int_image.shape[:2], dtype=numpy.uint32)
    image_view = rgba_image.view(
        dtype=numpy.uint8
    ).reshape(int_image.shape[:2] + (4, ))
    image_view[:, :, 3] = 255

    if int_image.ndim == 2:
        image_view[:, :, :3] = int_image[:, :, None]
    elif int_image.ndim == 3:
        image_view[:, :, :3] = int_image
    else:
        image_view = int_image

    return rgba_image


class Display2DData(BasePlotter):
    """Show images.

    Parameters
    ----------
    data_streams : fuel.streams.AbstractDataStream
        A list of streams
    """

    def __init__(self, data_streams, titles='', alpha=0.7, radius=0.1):
        self.data_streams = data_streams
        self.set_titles(titles, data_streams)
        self.alpha = alpha
        self.radius = radius
        for data_stream in data_streams:
            if 'features' not in data_stream.sources:
                raise ValueError("The data stream source does "
                                 "not include features.")

        super(Display2DData, self).__init__()

    def initialize(self):
        self.figs = []
        for data_stream, title in zip(self.data_streams, self.titles):
            self.figs.append(figure(title=title))

    def call(self):
        for fig, data_stream in zip(self.figs, self.data_streams):
            iterator = data_stream.get_epoch_iterator()
            fig.renderers = fig.renderers[:4]

            for data in iterator:
                coordinates = data[data_stream.sources.index('features')]
                if 'scores' in data_stream.sources:
                    color_intensities = data[
                        data_stream.sources.index('scores')]
                    color_intensities = numpy.round(
                        color_intensities[:, 0] * 255
                    )
                    colors = ["#%02x%02x%02x" % (255 - color_intensity,
                                                 color_intensity, 128)
                              for color_intensity in color_intensities]
                elif 'targets' in data_stream.sources:
                    color_intensities = data[
                        data_stream.sources.index('targets')]
                    color_intensities = numpy.round(
                        color_intensities[:, 0] * 255
                    )
                    colors = ["#%02x%02x%02x" % (255 - color_intensity,
                                                 color_intensity, 128)
                              for color_intensity in color_intensities]
                else:
                    colors = "#000000"

                fig.scatter(coordinates[:, 0], coordinates[:, 1],
                            radius=0.05, fill_color=colors, fill_alpha=1.,
                            line_color=None)

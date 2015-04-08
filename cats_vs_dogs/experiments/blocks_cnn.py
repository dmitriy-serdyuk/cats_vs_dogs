import logging
import os
import argparse
import yaml
import cPickle
import re
from collections import OrderedDict

import numpy

import theano
from theano import tensor

from blocks.algorithms import (GradientDescent, Scale, CompositeRule,
                               StepClipping, RMSProp)
from blocks.bricks import Softmax, Rectifier
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant, Uniform
from blocks.model import Model
from blocks.main_loop import MainLoop
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Dump
from blocks.extensions.training import SharedVariableModifier
from blocks.dump import load_parameter_values
from blocks.config_parser import Configuration
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph

from fuel.datasets import IterableDataset
from fuel.schemes import ConstantScheme, SequentialScheme, ShuffledScheme
from fuel.streams import BatchDataStream, DataStream
from fuel.transformers import Mapping, ForceFloatX

from cats_vs_dogs.iterators import (DogsVsCats, Unbatch,
                                    RandomCrop, Reshape,
                                    ImageTranspose, OneHotEncoder, GreySquare)
from cats_vs_dogs.bricks import ConvNN, Dropout
from cats_vs_dogs.algorithms import Adam
from cats_vs_dogs.schemes import SequentialShuffledScheme
from cats_vs_dogs.extensions import (ImageDataStreamDisplay, DisplayImage,
                                     PlotManager)

floatX = theano.config.floatX
logging.basicConfig(level='INFO')


def load_params(path, model):
    print path
    parameters = load_parameter_values(path + '/params.npz')
    with open(path + '/log', "rb") as source:
        log = cPickle.load(source)
    model.set_param_values(parameters)
    #main_loop.log = log


def parse_config(path):
    config = ConfigCats()
    config.add_config('image_shape', type_=int, default=221)
    config.add_config('scaled_size', type_=int, default=256)
    config.add_config('channels', type_=int, default=3)
    config.add_config('batch_size', type_=int, default=100)
    config.add_config('epochs', type_=int, default=50000)
    config.add_config('load', type_=bool, default=False)
    config.add_config('model_path', type_=str, default='./models/model')
    config.add_config('algorithm', type_=str, default=False)
    config.add_config('feature_maps', type_=list, default=[25, 50, 100])
    config.add_config('conv_sizes', type_=list, default=[7, 5, 3])
    config.add_config('pool_sizes', type_=list, default=[3, 3, 3])
    config.add_config('mlp_hiddens', type_=list, default=[500])
    config.add_config('learning_rate', type_=float, default=1.e-4)
    config.add_config('dropout', type_=bool, default=False)
    config.add_config('plot', type_=bool, default=False)
    config.add_config('test', type_=bool, default=False)
    config.load_yaml(path)
    return config


class ConfigCats(Configuration):
    def load_yaml(self, path):
        yaml_file = os.path.expanduser(path)
        if os.path.isfile(yaml_file):
            with open(yaml_file) as f:
                for key, value in yaml.safe_load(f).items():
                    if key not in self.config:
                        raise ValueError("Unrecognized config in YAML: {}"
                                         .format(key))
                    self.config[key]['yaml'] = value


def construct_stream(dataset, config, test=False, one_example=False):
    if test:
        scheme = None
    else:
        scheme = SequentialShuffledScheme(dataset.num_examples,
                                          config.batch_size, rng)
    if one_example:
        scheme = SequentialShuffledScheme(dataset.num_examples,
                                          1, rng)

    stream = DataStream(
        dataset=dataset,
        iteration_scheme=scheme)
    if not test:
        stream = OneHotEncoder(num_classes=2, data_stream=stream)
        stream = Unbatch(data_stream=stream)
    if not test:
        stream = Reshape(stream, 'X', 'shape')
    stream = RandomCrop(data_stream=stream, image_source='X',
                        crop_size=config.image_shape,
                        scaled_size=config.scaled_size, rng=rng)
    if test:
        batch_scheme = ConstantScheme(1)
    else:
        batch_scheme = ConstantScheme(config.batch_size)
    if one_example:
        batch_scheme = ConstantScheme(1)
    stream = BatchDataStream(data_stream=stream, iteration_scheme=batch_scheme)
    stream = ImageTranspose(stream, 'X')
    stream = GreySquare(stream, 'X')
    return stream


# Rescaling the data
class Rescale(object):
    def __init__(self, scale=1., shift=0.):
        self.scale = scale
        self.shift = shift

    def __call__(self, x):
        return (self.scale * (x[0] + self.shift),) + x[1:]


# Getting around having tuples as argument and output
class TupleMapping(object):
    def __init__(self, fn, same_len_out=False, same_len_in=False):
        self.fn = fn
        self.same_len_out = same_len_out
        self.same_len_in = same_len_in

    def __call__(self, args):
        if self.same_len_in:
            rval = (self.fn(*args), )
        else:
            rval = (self.fn(args[0]), )
        if self.same_len_out:
            rval += args[1:]
        return rval


if __name__ == '__main__':
    logging.info('.. starting')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()
    config = parse_config(args.config)

    conv_activations = [Rectifier() for _ in config.feature_maps]
    mlp_activations = [Rectifier() for _ in config.mlp_hiddens] + [None]
    convnet = ConvNN(conv_activations, config.channels,
                   (config.image_shape,) * 2,
                   filter_sizes=zip(config.conv_sizes, config.conv_sizes),
                   feature_maps=config.feature_maps,
                   pooling_sizes=zip(config.pool_sizes, config.pool_sizes),
                   top_mlp_activations=mlp_activations,
                   top_mlp_dims=config.mlp_hiddens + [2],
                   border_mode='full',
                   weights_init=IsotropicGaussian(0.1),
                   biases_init=Constant(0))
    convnet.initialize()

    x = tensor.tensor4('X')
    y = tensor.lmatrix('y')
    last_hidden = convnet.apply(x)
    cost = Softmax().categorical_cross_entropy(y, last_hidden)
    cost.name = 'cost'
    error_rate = MisclassificationRate().apply(tensor.argmax(y, axis=1),
                                               last_hidden)
    error_rate.name = 'error_rate'

    ouputs = [cost, error_rate]
    train_outputs = ouputs
    test_outputs = ouputs
    if config.dropout:
        dropout = Dropout(0.5, [x, y], ouputs)
        train_outputs = dropout.train_model()
        test_outputs = dropout.test_model()

    logging.info('.. model built')
    rng = numpy.random.RandomState(2014 + 02 + 04)
    train_dataset = DogsVsCats('train', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'))
    train_stream = construct_stream(train_dataset, config)
    from matplotlib import pyplot as plt
    #print 'start iter'
    #for data in train_stream.get_epoch_iterator():
    #    print 'start show'
    #    print data[0][0].shape
    #    plt.imshow(numpy.cast['uint8'](data[0][0].transpose(1, 2, 0) * 65. + 114.))
    #    plt.show()
    #    print 'end show'
    #    break

    test_dataset = DogsVsCats('test', os.path.join('${PYLEARN2_DATA_PATH}',
                                                   'dogs_vs_cats',
                                                   'train.h5'))
    test_stream = construct_stream(test_dataset, config)
    #for data in test_stream.get_epoch_iterator():
    #    plt.imshow(numpy.cast['uint8'](data[0][0].transpose(1, 2, 0) * 65. + 114.))
    #    plt.show()
    #    break

    #for i, data in enumerate(test_stream.get_epoch_iterator()):
    #    pass
    #print i
    valid_dataset = DogsVsCats('valid', os.path.join('${PYLEARN2_DATA_PATH}',
                                                     'dogs_vs_cats',
                                                     'train.h5'))
    valid_stream = construct_stream(valid_dataset, config, one_example=True)

    valid_monitor = DataStreamMonitoring(
        variables=test_outputs, data_stream=valid_stream, prefix="valid")
    test_monitor = DataStreamMonitoring(
        variables=test_outputs, data_stream=test_stream, prefix="test")

    extensions = []

    if config.algorithm == 'adam':
        step_rule = Adam()
    elif config.algorithm == 'rms_prop':
        step_rule = RMSProp(config.learning_rate)
    else:
        clipping = StepClipping(threshold=numpy.cast[floatX](100.))
        sgd = Scale(learning_rate=config.learning_rate)
        step_rule = CompositeRule([clipping, sgd])
        adjust_learning_rate = SharedVariableModifier(
            sgd.learning_rate,
            lambda n: 10. / (10. / config.learning_rate + n))
        extensions += [adjust_learning_rate]
    model = Model(train_outputs[0])
    learn_params = OrderedDict([(name, param) for name, param in model.get_params().items()
                                if not re.match('.*conv_pool_[0-4].*', name)
                                ])
    print 'learn_parameters', learn_params
    algorithm = GradientDescent(cost=train_outputs[0], step_rule=step_rule,
                                params=learn_params.values())
    train_monitor = TrainingDataMonitoring(
        variables=train_outputs + [
                   aggregation.mean(algorithm.total_gradient_norm)],
        prefix="train", after_epoch=True)
    extensions += [FinishAfter(after_n_epochs=config.epochs),
                   train_monitor,
                   valid_monitor,
                   test_monitor,
                   ProgressBar(),
                   Printing(),
                   Dump(config.model_path, after_epoch=True)
                   ]
    print train_monitor.record_name(cost)
    print valid_monitor.record_name(cost)
    if config.plot:
        extensions += [Plot(os.path.basename(config.model_path),
                            [[train_monitor.record_name(cost),
                              valid_monitor.record_name(cost),
                              test_monitor.record_name(cost)],
                             [train_monitor.record_name(error_rate),
                              valid_monitor.record_name(error_rate),
                              test_monitor.record_name(error_rate)]])]
    # Deconvolution
    num_filters = 10
    x_repeated = x.repeat(num_filters, axis=0)
    out = convnet.apply(x_repeated)
    cg_repeat = ComputationGraph(out)
    convnet_features_repeated, = VariableFilter(applications=[convnet.layers[0].apply], name='output')(cg_repeat.variables)
    convnet_features_selected = convnet_features_repeated \
        * tensor.eye(num_filters).repeat(
            x.shape[0], axis=0
        ).dimshuffle((0, 1, 'x', 'x'))
    displayable_deconvolution = tensor.grad(error_rate, x_repeated,
                                            known_grads={
                                                convnet_features_selected:
                                                    convnet_features_selected
                                            })
    deconvolution_normalizer = abs(
        displayable_deconvolution
    ).max(axis=(1, 2, 3))
    displayable_deconvolution = displayable_deconvolution \
        / deconvolution_normalizer.dimshuffle((0, 'x', 'x', 'x'))
    get_displayable_deconvolution = theano.function(
        [x], displayable_deconvolution
    )
    one_example_train_data_stream = construct_stream(train_dataset, config, one_example=True)
    display_deconvolution_data_stream = Mapping(
        data_stream=one_example_train_data_stream,
        mapping=TupleMapping(get_displayable_deconvolution,
                             same_len_out=True)
    )
    display_deconvolution = ImageDataStreamDisplay(
        data_stream=display_deconvolution_data_stream,
        source='X',
        image_shape=(3, 32, 32),
        axes=('c', 0, 1),
        shift=0,
        rescale=1.,
    )

    images_displayer = DisplayImage(
        image_getters=[display_deconvolution],
        titles=['Deconvolution']
    )
    plotters = []
    plotters.append(images_displayer)

    from scipy import misc
    from matplotlib import pyplot as plt
    if config.load:
        load_params(config.model_path, model)

    extensions.append(PlotManager('2 layer deconvolution',
                                  plotters=plotters,
                                  after_epoch=False,
                                  every_n_epochs=10,
                                  after_training=True))

    def visualise(unit, layer):
        path = '/data/lisatmp3/serdyuk/catsvsdogs/test1/'
        input_initial = rng.normal(0, 0.1, (1, 3, 280, 280))
        #input_initial = numpy.array([(misc.imread(path + '1.jpg').transpose(2, 0, 1) - 114.) / 65.])[:, :, :280, :280]
        #input_initial = numpy.zeros((1, 3, 280, 280))
        x_vis = theano.shared(input_initial, 'x_vis')
        last_hidden_vis = convnet.apply(x_vis)
        cg_vis = ComputationGraph(last_hidden_vis)

        out, = VariableFilter(applications=[convnet.layers[layer].apply],
                              name='output')(cg_vis.variables)
        #print out.ndim
        #print theano.function([], out)().shape

        out = out[0, unit, 0, 0]
        #out = Softmax().apply(last_hidden_vis)[0, 0]
        #out = last_hidden_vis[0, 1] #/ last_hidden_vis.sum()
        gradient_1 = tensor.grad(out - 0.1 * (x_vis ** 2).sum()
                                , x_vis)
        lr = theano.shared(1.e-1)
        updates = {x_vis: x_vis + lr * gradient_1}
        updates[lr] = lr * 0.999
        make_step = theano.function([], [gradient_1, out], updates=updates)
        compute_prob = theano.function([x], last_hidden)

        for i in xrange(600):
            grad, val = make_step()
            print lr.get_value()
            print 'step', i, 'val', val

        max_val = grad#.get_value()
        std = max_val.std()
        mean = max_val.mean()
        print max_val[0].transpose(1, 2, 0)
        print compute_prob(x_vis.get_value())
        print grad
        #max_val = (max_val[0].transpose(1, 2, 0) - mean) / std * 65. + 128.
        #plt.imshow(numpy.cast['uint8'](max_val))
        #plt.show()
        val = x_vis.get_value()[0]#[0, :, 0:10, :10]
        return val, (max_val[0] - mean) / std

    if True:
        print 'start comp'
        compute_prob = theano.function([x], Softmax().apply(last_hidden))
        print 'start comp'
        for data in valid_stream.get_epoch_iterator():
            image = numpy.zeros_like(data[0])
            probs_all = []
            for i in xrange(28 * 28):
                print i
                print data[0][i * 100:(i + 1) * 100, 0, :, :, :].shape
                #probs = compute_prob(data[0][i * 100:(i + 1) * 100, 0, :, :, :])
                #probs_all.append(probs)
            probs_all = numpy.concatenate(probs_all, axis=0)
            print probs_all.shape
            image = probs_all[:, 0].reshape((280, 280))
            assert False
        vals = []
        grads = []
        for i in xrange(1):
            val, grad = visualise(11, 2)
            vals.append(val)
            grads.append(grad)

        for i in xrange(10):
            plt.imshow(numpy.cast['uint8'](vals[i].transpose(1, 2, 0) * 65. + 128.),
                       interpolation='none')
            plt.show()
            plt.imshow(numpy.cast['uint8'](grads[i].transpose(1, 2, 0) * 65. + 128.),
                       interpolation='none')
            plt.show()
        #numpy.save('1layer_layer0_maps.npy', vals)
        assert False
        path = '/data/lisatmp3/serdyuk/catsvsdogs/test1/'
        predict = theano.function([x], [last_hidden, Softmax().apply(last_hidden)])
        for filename in os.listdir(path):
            if not os.path.isfile(path + '/' + filename):
                continue
            image = misc.imread(path + '/' + filename)
            dataset = IterableDataset({'X': [image, image, image, image, image]})
            stream = construct_stream(dataset, config, test=True)

            preds = []
            for data in stream.get_epoch_iterator():
                pred = predict(data[0])
                preds.append(pred[0].argmax())
            print '%s, %d' % (filename[:-4], 1 if sum(preds) > 2 else 0)
        assert False

    main_loop = MainLoop(model=model, data_stream=train_stream,
                         algorithm=algorithm, extensions=extensions)
    main_loop.run()

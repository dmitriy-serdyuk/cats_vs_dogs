__author__ = 'serdyuk'

import logging
import os
import inspect

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.dump import MainLoopDumpManager, inject_parameter_values
from blocks.utils import reraise_as

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


class SharedVariableModifier(TrainingExtension):
        """Adjusts shared variable parameter using some function.

        Applies function to compute the new value of a shared parameter each
        iteration.

        This class can be used to adapt over the training process parameters like
        learning rate, momentum, etc.

        Parameters
        ----------
        parameter : :class:`~tensor.TensorSharedVariable`
            shared variable to be adjusted
        function : callable
            a function which outputs a numeric value to which the
            given shared variable will be set and may take one or two arguments.

            In the first case, function that takes the total number of examples
            seen (``int``) as an input.

            In the second case, it is a function which takes number of examples
            seen (``int``) and old value of the shared variable.

        """
        def __init__(self, parameter, function, **kwargs):
            super(SharedVariableModifier, self).__init__(**kwargs)
            self.parameter = parameter
            self.function = function
            self.num_examples = 0
            self.num_args = len(inspect.getargspec(function).args)

        def after_batch(self, batch):
            self.num_examples += batch.values()[0].shape[0]
            if self.num_args == 1:
                new_value = self.function(self.num_examples)
            else:
                old_value = self.parameter.get_value()
                new_value = self.function(self.num_examples, old_value)
            self.parameter.set_value(new_value)


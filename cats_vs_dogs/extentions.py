__author__ = 'serdyuk'

import logging
import os

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


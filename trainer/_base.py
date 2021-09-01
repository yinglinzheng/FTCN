#!/usr/bin/python
# -*- coding: UTF-8 -*-


import time
from abc import ABC,abstractmethod
from utils import logger
from config import config as cfg
import os
import sys
import torch.distributed as dist

class TrainerBase(ABC):

    @property
    def timestamp(self):
        """ Standardised timestamp for loss reporting """
        return time.strftime("%H:%M:%S")


    # you should include training logic in run
    @abstractmethod
    def run(self):
        pass



    @property
    def name(self):
        """ Set the model name based on the subclass """
        basename = os.path.basename(sys.modules[self.__module__].__file__)
        retval = os.path.splitext(basename)[0].lower()
        logger.debug("trainer name: '%s'", retval)
        return retval

    @property
    def config(self) -> dict:
        """
        Return private config dict, to distinguish with orgin config this is in type
        dictionary,the specific config will overide default config for current plugin

        """
        if '_config' not in self.__dict__:
            model_type, model_name = self.__module__.split(".")[-2:]
            logger.debug("Loading config for: %s", model_name)
            trainer_config = getattr(cfg, 'trainer',{}).to_dict()
            default_config = trainer_config.get('default', {})
            specific_config = trainer_config.get(model_name, {})
            self._config = {}
            for k,v in default_config.items():
                self._config[k] = v
            for k,v in specific_config.items():
                self._config[k] = v
        return self._config

    def check_load(self, *loaed_states):
        if not all(loaed_states) and any(loaed_states):
            logger.warn('model is partially loaded this could lead to '
                        'unexpected situation ')

    @property
    def sample_dir(self):
        return os.path.join(cfg.path.model_dir, cfg.setting_name, 'sample')

    @property
    def rank(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        return rank

    @property
    def is_distributed(self):
        return dist.is_initialized()

    @property
    def worldsize(self):
        return dist.get_world_size() if self.is_distributed else 1

    def save_all(self, ep):
        for model in self.need_to_save:
            model.save_models(ep)

    def load_all(self):
        flags = []
        epochs = []
        for model in self.need_to_load:
            flag, epoch = model.load()
            flags.append(flag)
            epochs.append(epoch)
        self.check_load(flags)
        if len(set(epochs)) != 1:
            logger.warn('model is loaded from different epoch')
        return epochs[0]

    def apply_all(self, object_list, method, *args,**kwargs):
        return [getattr(obj, method)(obj, *args,**kwargs) for obj in object_list]

    def __getattr__(self, name):
        if name not in self.__dict__:
            v = self.config.get(name,None)
            if v is not None:
                logger.warn('{} has no attribute {},but found in it\'s private config, '
                            'this is not the recomendation way.'.format(self.name, name))
                return v
            raise AttributeError(name)
        else:
            return getattr(self, name)

    def asign_writer(self, writer):
        self.writer = writer
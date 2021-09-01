#!/usr/bin/python
# -*- coding: UTF-8 -*-



from utils import logger
from config import config as cfg
import os
import sys
import glob
import torch
from torch import nn
from abc import ABC,abstractmethod
import torch.distributed as dist
from config import config,AttrDict
from typing import Callable
import inspect
import traceback
import functools
from typing import Type

#try:
#    from apex.parallel import DistributedDataParallel as DDP
#    from apex.fp16_utils import *
#    from apex import amp, optimizers
#    from apex.multi_tensor_apply import multi_tensor_applier

#except ImportError:
#    raise ImportError("Model v2 are need apex and inplaceabn as a prerequsite")


class ModelBase(nn.Module):
    """ Base class that all models should inherit from """

    def __init__(self):
        super().__init__()
        self.network = self.build_network()
        self._warped_network = self.network

    def forward(self, *input, **kwargs):
       return self._warped_network(*input, **kwargs)

    def save_models(self, epoch):
        """ Backup and save the models """
        if self.rank ==0:
            logger.debug("Backing up and saving models")
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)
            torch.save(self.network.state_dict(), self.get_checkpoint_path(epoch))
            if os.path.exists(self.get_checkpoint_path(epoch - self.max_to_keep)):
                os.remove(self.get_checkpoint_path(epoch - self.max_to_keep))
            logger.info("{} models saved".format(self.name))

    def load(self, fullpath=None, epoch=-1,pretrained=None):
        """ Force Loading a model, or load the latest model"""
        if fullpath is None:
            fullpath, loaded_epoch = self.find_last(epoch)
        else:
            loaded_epoch = epoch
        if fullpath is None:
            if pretrained is None:
                logger.info("No existing {} model found".format(self.name))
                return False, -1
            else:
                logger.info("pretrained {} model found".format(self.name))
                fullpath=pretrained
                loaded_epoch=-1

        logger.debug("Loading model: '%s'", fullpath)
        try:
            saved_state_dict = torch.load(fullpath, map_location='cpu')


            

            param_dict = self.network.state_dict()
            all_key = set(param_dict.keys())
            unmatch_key = set()
            loaded_key = set()
            redundant_key = set()
            for k, v in saved_state_dict.items():
                
                # for compatibilty issue
                if k.startswith('module'):
                    k = k[7:]

                if k not in param_dict:
                    redundant_key.add(k)
                    continue
                if param_dict[k].size() != v.size():
                    unmatch_key.add(k)
                    continue
                param_dict[k] = v
                loaded_key.add(k)
            unfind_key = all_key - (loaded_key | unmatch_key)
            if redundant_key:
                logger.warn("{} are in checkpoint, but not found in model {}".format(redundant_key, self.name))
            if unfind_key:
                logger.warn(
                    "{} are in model, but not found in chekppint loaded for {}".format(unfind_key, self.name))
            if unmatch_key:
                logger.warn(
                    "{} have unmatching shape between checkpoint&model for {}".format(unmatch_key, self.name))
            self.network.load_state_dict(param_dict)
            logger.info(" consume training from {}".format(fullpath))
        except ValueError as err:
            logger.warning("Failed loading existing training data for {}. Generating new models".format(self.name))
            logger.debug("Exception: %s", str(err))
            return False, -1
        except OSError as err:
            logger.warning("Failed loading existing training data for {}. Generating new models".format(self.name))
            logger.debug("Exception: %s", str(err))
            return False, -1
        except:
            logger.error(traceback.format_exc())
            raise
        return True, loaded_epoch

    def get_checkpoint_path(self, epoch):
        """" returning the checkpoint path  w.r.t epoch  which should be {name}_{epoch}.pth"""
        return os.path.join(self.model_dir, self.name + '_' + str(epoch) + '.pth')

    def make_distributed(self, distributed=True):
        """"this will make_distributed"""
        if dist.is_initialized() and distributed:
            #self._warped_network = DDP(self.network,
            #                 #device_ids=[torch.cuda.current_device()],
            #                 #output_device=torch.cuda.current_device(),
            #                 delay_allreduce=True,
            #                 )
            self._warped_network = torch.nn.parallel.DistributedDataParallel(self._warped_network,
                                                                             device_ids=[torch.cuda.current_device()],
                                                                             output_device=torch.cuda.current_device(),
                                                                             find_unused_parameters=True
                                                                             )
        else:
            logger.warning("running in non distributed mode")


    # TODO remove the compatibility  property
    #@property
    #def _warped_network(self):
    #    return self.network


    @property
    def rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def max_to_keep(self):
        return cfg.max_to_keep

    def build_network(self) -> nn.Module:
        """"you must build and return an nn.Module here"""
        return self.module_to_build(**self.kwargs_for_build)

    @property
    def must_kwargs(self) -> list:
        """"the config key must be past to build method"""
        sig = inspect.signature(self.module_to_build.__init__).parameters
        return [k for k,v in sig.items() if v.default is inspect._empty and k != 'self']

    @property
    def optional_kwargs(self) -> list:
        """"the optional key could be past to build method"""
        sig = inspect.signature(self.module_to_build.__init__).parameters
        return [k for k, v in sig.items() if v.default is not inspect._empty and k != 'self']

    @property
    def module_to_build(self) -> Type[nn.Module]:
        """"this function which kwargs are feed to"""
        pass

    @property
    def kwargs_for_build(self):
        kwargs = {}
        missing_keys = []
        for k in self.must_kwargs:
            if k in self.config:
                kwargs[k] = self.config[k]
            else:
                missing_keys.append(k)
        assert not missing_keys, 'key: {} must be parsed for building {}'.format(missing_keys,self.name)

        for k in self.optional_kwargs:
            if k in self.config:
                kwargs[k] = self.config[k]

        return kwargs

    def freeze(self):
        """"you must state if this model need to be train"""
        for param in self.parameters():
            param.requires_grad = False

    # for better compatibilty with the model v1
    def parameters(self, recurse=True):
        return self.network.parameters(recurse)

    @property
    def model_dir(self):
        return os.path.join(cfg.path.model_dir, cfg.setting_name)

    @property
    def name(self):
        """ Set the model name based on the subclass """
        basename = os.path.basename(sys.modules[self.__module__].__file__)
        retval = os.path.splitext(basename)[0].lower()
        logger.debug("model name: '%s'", retval)
        return retval


    def find_last(self, epoch=-1, model_dir=None):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            checkpoint :The path of the last checkpoint file
        """
        if model_dir is None:
            model_dir = self.model_dir

        if not os.path.exists(model_dir):
            logger.info("model dir not exists {} ".format(model_dir))
            return None, -1
        #assert os.path.exists(self.model_dir), "model dir not exists {}".format(self.model_dir)

        checkpoints = glob.glob(os.path.join(model_dir, '*.pth'))

        checkpoints = list(filter(lambda x: os.path.basename(x).startswith(self.name), checkpoints))
        if len(checkpoints) == 0:
            return None, -1
        checkpoints = {int(os.path.basename(x).split('.')[0].split('_')[-1]):x for x in checkpoints}

        start = min(checkpoints.keys())
        end = max(checkpoints.keys())

        if epoch == -1:
            return checkpoints[end], end
        elif epoch < start :
            raise RuntimeError(
                "model for epoch {} has been deleted as we only keep {} models".format(epoch,self.max_to_keep))
        elif epoch > end:
            raise RuntimeError(
                "epoch {} is bigger than all exist checkpoints".format(epoch))
        else:
            return checkpoints[epoch], epoch

    @property
    def config(self) -> dict:
        """
        Return private config dict, to distinguish with origin config this is in type
        dictionary,the specific config will overide default config for current plugin
        """
        if '_config' not in self.__dict__:
            model_type, model_name = self.__module__.split(".")[-2:]
            logger.debug("Loading config for: %s", model_name)
            model_config = getattr(config,model_type,AttrDict()).to_dict()
            default_config = model_config.get('default', {})
            specific_config = model_config.get(model_name, {})
            self._config = {}
            for k, v in config.to_dict().items():
                if not isinstance(v,dict):
                    self._config[k] = v
            for k,v in default_config.items():
                self._config[k] = v
            for k,v in specific_config.items():
                self._config[k] = v
        return self._config

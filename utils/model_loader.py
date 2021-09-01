#!/usr/bin/python
# -*- coding: UTF-8 -*-


import types
from utils import logger
from config import config as cfg
import os
import sys
import glob
import torch
import traceback
import types
import torch.distributed as dist
import copy
from .torch_save import torch_save

def add_loader(target, name,max_to_keep=2):

    def get_rank(self):
        return dist.get_rank() if dist.is_initialized() else 0

    def save_models(self, epoch):
        """ Backup and save the models """
        if self.get_rank() == 0:
            logger.debug("Backing up and saving models")
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)
            
            torch_save(self.state_dict(), self.get_checkpoint_path(epoch))
            if os.path.exists(self.get_checkpoint_path(epoch - self.max_to_keep)):
                os.remove(self.get_checkpoint_path(epoch - self.max_to_keep))
            logger.info("{} models saved".format(self.name))

    def load(self, fullpath=None, epoch=-1):
        """ Force Loading a model, or load the latest model"""
        if fullpath is None:
            fullpath, loaded_epoch = self.find_last(epoch)
        else:
            loaded_epoch = epoch

        if fullpath is None:
            logger.info("No existing {} model found".format(self.name))
            return False, -1
        logger.debug("Loading model: '%s'", fullpath)
        try:
            saved_state_dict = torch.load(fullpath, map_location='cpu')
            self.load_state_dict(saved_state_dict)
            logger.info(" consume training from {}".format(fullpath))
        except ValueError as err:
            logger.warning("Failed loading existing training data for {}. Generating new models".format(self.name))
            logger.debug("Exception: %s", str(err))
            return False, -1
        except OSError as err:
            logger.warning("Failed loading existing training data for {}. Generating new models".format(self.name))
            logger.debug("Exception: %s", str(err))
            return False, -1
        except RuntimeError as err:
            logger.warning("{} model has corrupted, try to load earlier one".format(self.name))
            logger.debug("Exception: %s", str(err))
            return False, -1
        except:
            logger.error(traceback.format_exc())
            raise

        return True, loaded_epoch

    def get_checkpoint_path(self, epoch):
        """" returning the checkpoint path  w.r.t epoch  which should be {name}_{epoch}.pth"""
        return os.path.join(self.model_dir, self.name + '_' +str(epoch) + '.pth')


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

    target.find_last = types.MethodType(find_last, target)
    target.get_checkpoint_path = types.MethodType(get_checkpoint_path, target)
    target.load = types.MethodType(load, target)
    target.save_models = types.MethodType(save_models, target)
    target.get_rank = types.MethodType(get_rank, target)

    target.max_to_keep = max_to_keep
    target.name = name
    target.model_dir = os.path.join(cfg.path.model_dir, cfg.setting_name)
    return target
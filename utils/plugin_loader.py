#!/usr/bin/python
# -*- coding: UTF-8 -*-



""" Plugin loader for extract, training and model tasks """

from utils import logger
import os
from importlib import import_module
from typing import Type
from trainer._base import TrainerBase
from torch.utils.data import Dataset
from model._base import ModelBase

class PluginLoader():
    """
    Plugin loader for extract, training and model tasks
    function: get_{model_type}
    args: {model_name}
    will return the a class named model_type under  model_type.model_name.py

    as it return a class you should also annotate the returning classtype to make
    code linting avaliable in some IDE
    """
    @staticmethod
    def get_classifier(name) -> Type[ModelBase]:
        """ Return requested attribute encoder plugin """
        return PluginLoader._import("model.classifier", name)

    @staticmethod
    def get_trainer(name) -> Type[TrainerBase]:
        """ Return requested trainer plugin """
        return PluginLoader._import("trainer", name)

    @staticmethod
    def get_dataset(name) -> Type[Dataset]:
        """ Return requested trainer plugin """
        return PluginLoader._import("dataset", name)

    @staticmethod
    def _import(attr, name):
        """ Import the plugin's module """
        name = name.replace("-", "_")
        ttl = attr.split(".")[-1].title()
        logger.info("Loading %s from %s plugin...", ttl, name.title())
        attr = "model" if attr == "Trainer" else attr.lower()
        mod = ".".join((attr, name))
        module = import_module(mod)
        logger.info(str(module) + str(ttl))
        return getattr(module, ttl)

    @staticmethod
    def get_available_trainer():
        """ Return a list of available models """
        modelpath = os.path.join(os.path.dirname(__file__), "trainer")
        models = sorted(item.name.replace(".py", "").replace("_", "-")
                        for item in os.scandir(modelpath)
                        if not item.name.startswith("_")
                        and item.name.endswith(".py"))
        return models

    @staticmethod
    def get_default_model():
        """ Return the default model """
        models = PluginLoader.get_available_models()
        return 'original' if 'original' in models else models[0]



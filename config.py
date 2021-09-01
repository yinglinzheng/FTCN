#!/usr/bin/python
# Borrow from tensorpack,credits goes to yuxin wu

# the loaded sequnce is
#   default config in this file
#   -> provided setting file  (you can not add new config after this)
#   -> manully overrided config
#   -> computed config in finalize config (you can change config after this)

import os
import pprint
import yaml

__all__ = ["config", "finalize_configs"]


class AttrDict:
    _freezed = False
    """ Avoid accidental creation of new hierarchies. """

    def __getattr__(self, name):
        if self._freezed:
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name, value):
        if self._freezed and name not in self.__dict__:
            raise AttributeError("Cannot create new attribute!")
        super().__setattr__(name, value)

    def __str__(self):
        return pprint.pformat(self.to_dict(), indent=1)

    __repr__ = __str__

    def to_dict(self):
        """Convert to a nested dict. """
        return {
            k: v.to_dict() if isinstance(v, AttrDict) else v
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def update_args(self, args):
        """Update from command line args. """
        for cfg in args:
            keys, v = cfg.split("=", maxsplit=1)
            keylist = keys.split(".")
            dic = self
            # print(keylist)
            if len(keylist) == 1:
                assert keylist[0] in dir(dic), "Unknown config key: {}".format(
                    keylist[0]
                )
            for i, k in enumerate(keylist[:-1]):
                assert k in dir(dic), "Unknown config key: {}".format(k)
                dic = getattr(dic, k)
            key = keylist[-1]
            assert key in dir(dic), "Unknown config key: {}".format(key)
            oldv = getattr(dic, key)
            if not isinstance(oldv, str):
                v = eval(v)
            setattr(dic, key, v)

    def update_with_yaml(self, rel_path):
        base_path = os.path.dirname(os.path.abspath(__file__))
        setting_path = os.path.normpath(os.path.join(base_path, "setting", rel_path))
        setting_name = os.path.basename(setting_path).split(".")[0]

        with open(setting_path, "r") as f:
            overrided_setting = yaml.load(f,Loader=yaml.FullLoader)
            # if  'setting_name' not in overrided_setting:
            #    raise RuntimeError('you must provide a setting name for non root_setting: {}'.format(rel_path))
            self.update_with_dict(overrided_setting)
        setattr(self, "setting_name", setting_name)

    def init_with_yaml(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        setting_path = os.path.normpath(os.path.join(base_path, "root_setting.yaml"))
        with open(setting_path, "r") as f:
            overrided_setting = yaml.load(f,Loader=yaml.FullLoader)
            self.update_with_dict(overrided_setting)

    def update_with_text(self,text):
        overrided_setting = yaml.load(text, Loader=yaml.FullLoader)
        self.update_with_dict(overrided_setting)

    def update_with_dict(self, dicts):
        for k, v in dicts.items():
            if isinstance(v, dict):
                getattr(self, k).update_with_dict(v)
            else:
                setattr(self, k, v)

    def freeze(self):
        self._freezed = True
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.freeze()

    # avoid silent bugs
    def __eq__(self, _):
        raise NotImplementedError()

    def __ne__(self, _):
        raise NotImplementedError()


config = AttrDict()
_C = config  # short alias to avoid coding


# you can directly write setting here as _C.model_dir='.\checkpoint' or in root_setting.yaml


#


def finalize_configs(input_cfg=_C, freeze=True, verbose=True):

    # _C.base_path = os.path.dirname(os.path.abspath(__file__))
    input_cfg.base_path = os.path.dirname(__file__)

    # for running in remote server
    # for k, v in input_cfg.path.__dict__.items():
    #     v = os.path.normpath(os.path.join(input_cfg.base_path, v))
    #     setattr(input_cfg.path, k, v)
    if freeze:
        input_cfg.freeze()
    # if verbose:
    #   logger.info("Config: ------------------------------------------\n" + str(_C))


if __name__ == "__main__":
    print("?")
    print(os.path.dirname(__file__))

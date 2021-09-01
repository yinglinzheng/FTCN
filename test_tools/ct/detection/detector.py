import os

import numpy as np
import torch

from .alignment import load_net, batch_detect


def get_project_dir():
    current_path = os.path.abspath(os.path.join(__file__, "../"))
    return current_path


def relative(path):
    path = os.path.join(get_project_dir(), path)
    return os.path.abspath(path)


class RetinaFace:
    def __init__(
        self, gpu_id=-1, model_path=None, network="mobilenet",
    ):
        self.gpu_id = gpu_id
        self.device = (
            torch.device("cpu") if gpu_id == -1 else torch.device("cuda", gpu_id)
        )
        self.model = load_net(model_path, self.device, network)

    def detect(self, images):
        if isinstance(images, np.ndarray):
            if len(images.shape) == 3:
                return batch_detect(self.model, [images], self.device)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, self.device)
        elif isinstance(images, list):
            return batch_detect(self.model, np.array(images), self.device)
        elif isinstance(images, torch.Tensor):
            if len(images.shape) == 3:
                return batch_detect(self.model, images.unsqueeze(0), self.device)[0]
            elif len(images.shape) == 4:
                return batch_detect(self.model, images, self.device)
        else:
            raise NotImplementedError()

    def __call__(self, images):
        return self.detect(images)

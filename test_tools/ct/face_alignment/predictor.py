# Face alignment demo
# Uses MTCNN as face detector
# Cunjian Chen (ccunjian@gmail.com)
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader
from .basenet import MobileNet_GDConv


def get_device(gpu_id):
    if gpu_id > -1:
        return torch.device(f"cuda:{str(gpu_id)}")
    else:
        return torch.device("cpu")


def load_model(file):
    model = MobileNet_GDConv(136)
    if file is not None:
        model.load_state_dict(torch.load(file, map_location="cpu"))
    else:
        url = "https://github.com/yinglinzheng/face_weights/releases/download/v1/mobilenet_224_model_best_gdconv_external.pth"
        model.load_state_dict(torch.utils.model_zoo.load_url(url))
    return model


# landmark of (5L, 2L) from [0,1] to real range
def reproject(bbox, landmark):
    landmark_ = landmark.clone()
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    landmark_[:, 0] *= w
    landmark_[:, 0] += x1
    landmark_[:, 1] *= h
    landmark_[:, 1] += y1
    return landmark_


def prepare_feed(img, face):
    height, width, _ = img.shape
    mean = np.asarray([0.485, 0.456, 0.406])
    std = np.asarray([0.229, 0.224, 0.225])
    out_size = 224
    x1, y1, x2, y2 = face[:4]

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(min([w, h]) * 1.2)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = torch.Tensor([x1, y1, x2, y2]).int()
    x1, y1, x2, y2 = new_bbox
    cropped = img[y1:y2, x1:x2]
    if dx > 0 or dy > 0 or edx > 0 or edy > 0:
        cropped = cv2.copyMakeBorder(
            cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0
        )
    cropped_face = cv2.resize(cropped, (out_size, out_size))

    if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
        return None
    test_face = cropped_face.copy()
    test_face = test_face / 255.0
    test_face = (test_face - mean) / std
    test_face = test_face.transpose((2, 0, 1))
    data = torch.from_numpy(test_face).float()
    return dict(data=data, bbox=new_bbox)


@torch.no_grad()
def single_predict(model, feed, device):
    landmark = model(feed["data"].unsqueeze(0).to(device)).cpu()
    landmark = landmark.reshape(-1, 2)
    landmark = reproject(feed["bbox"], landmark)
    return landmark.numpy()


@torch.no_grad()
def batch_predict(model, feeds, device):
    if not isinstance(feeds, list):
        feeds = [feeds]
    # loader = DataLoader(FeedDataset(feeds), batch_size=50, shuffle=False)
    data = []
    for feed in feeds:
        data.append(feed["data"].unsqueeze(0))
    data = torch.cat(data, 0).to(device)
    results = []

    landmarks = model(data).cpu()
    for landmark, feed in zip(landmarks, feeds):
        landmark = landmark.reshape(-1, 2)
        landmark = reproject(feed["bbox"], landmark)
        results.append(landmark.numpy())
    return results


@torch.no_grad()
def batch_predict2(model, feeds, device, batch_size=None):
    if not isinstance(feeds, list):
        feeds = [feeds]
    if batch_size is None:
        batch_size = len(feeds)
    loader = DataLoader(feeds, batch_size=len(feeds), shuffle=False)
    results = []
    for feed in loader:
        landmarks = model(feed["data"].to(device)).cpu()
        for landmark, bbox in zip(landmarks, feed["bbox"]):
            landmark = landmark.reshape(-1, 2)
            landmark = reproject(bbox, landmark)
            results.append(landmark.numpy())
    return results


class LandmarkPredictor:
    def __init__(self, gpu_id=0, file=None):
        self.device = get_device(gpu_id)
        self.model = load_model(file).to(self.device).eval()

    def __call__(self, feeds):
        results = batch_predict2(self.model, feeds, self.device)
        if not isinstance(feeds, list):
            results = results[0]
        return results

    @staticmethod
    def prepare_feed(img, face):
        return prepare_feed(img, face)

import cv2
from .detector import RetinaFace
from .utils import *


def assert_bounded(val, low, up):
    return val >= low and val < up


def check_valid(face, w, h):
    box = face[0]
    if box[0] > box[2]:
        return False
    if box[1] > box[3]:
        return False
    for idx, bound in zip([0, 1, 2, 3], [w, h, w, h]):
        if not assert_bounded(box[idx], 0, bound):
            return False
    pts = face[1]
    for p in pts:
        for idx, bound in zip([0, 1], [w, h]):
            if not assert_bounded(p[idx], 0, bound):
                return False
    return True


def post_detect(detect_results, scale, w, h):
    new_results = []
    for frame_faces in detect_results:
        new_frame_faces = []
        for box, ldm, score in frame_faces:
            box = box * scale
            ldm = ldm * scale
            face = (box, ldm, score)
            if check_valid(face, w=w, h=h):
                new_frame_faces.append(face)
        new_results.append(new_frame_faces)
    return new_results


class FaceDetector(RetinaFace):
    def scale_detect(self, images):
        max_res = 1920
        h, w = images[0].shape[:2]
        if max(h, w) > max_res:
            init_scale = max(h, w) / max_res
        else:
            init_scale = 1
        resize_scale = 2 * init_scale
        resize_w = int(w / resize_scale)
        resize_h = int(h / resize_scale)
        detect_input = [cv2.resize(frame, (resize_w, resize_h)) for frame in images]
        detect_results = post_detect(
            self.detect(detect_input), scale=resize_scale, w=w, h=h,
        )
        return detect_results

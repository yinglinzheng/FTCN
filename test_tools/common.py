import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .ct.detection.utils import grab_all_frames, get_valid_faces, sample_chunks
from .ct.operations import multiple_tracking
import numpy as np
from .ct.face_alignment import LandmarkPredictor
from .ct.detection import FaceDetector
import cv2
from .utils import flatten,partition


detector = FaceDetector(0)
predictor = LandmarkPredictor(0)


def get_five(ldm68):
    groups = [range(36, 42), range(42, 48), [30], [48], [54]]
    points = []
    for group in groups:
        points.append(ldm68[group].mean(0))
    return np.array(points)


def get_bbox(mask):
    try:
        y, x = np.nonzero(mask[..., 0])
        return x.min() - 1, y.min() - 1, x.max() + 1, y.max() + 1
    except:
        return None


def get_bigger_box(image, box, scale=0.5):
    height, width = image.shape[:2]
    box = np.rint(box).astype(np.int)
    new_box = box.reshape(2, 2)
    size = new_box[1] - new_box[0]
    diff = scale * size
    diff = diff[None, :] * np.array([-1, 1])[:, None]
    new_box = new_box + diff
    new_box[:, 0] = np.clip(new_box[:, 0], 0, width - 1)
    new_box[:, 1] = np.clip(new_box[:, 1], 0, height - 1)
    new_box = np.rint(new_box).astype(np.int)
    return new_box.reshape(-1)


def process_bigger_clips(clips, dete_res, clip_size, step, scale=0.5):
    assert len(clips) % clip_size == 0
    detect_results = sample_chunks(dete_res, clip_size, step)
    clips = sample_chunks(clips, clip_size, step)
    new_clips = []
    for i, (frame_clip, record_clip) in enumerate(zip(clips, detect_results)):
        tracks = multiple_tracking(record_clip)
        for j, track in enumerate(tracks):
            new_images = []
            for (box, ldm, _), frame in zip(track, frame_clip):
                big_box = get_bigger_box(frame, box, scale)
                x1, y1, x2, y2 = big_box
                top_left = big_box[:2][None, :]
                new_ldm5 = ldm - top_left
                box = np.rint(box).astype(np.int)
                new_box = (box.reshape(2, 2) - top_left).reshape(-1)
                feed = LandmarkPredictor.prepare_feed(frame, box)
                ldm68 = predictor(feed) - top_left
                new_images.append(
                    (frame[y1:y2, x1:x2], big_box, new_box, new_ldm5, ldm68)
                )
            new_clips.append(new_images)
    return new_clips


def post(detected_faces):
    return [[face[:4], None, face[-1]] for face in detected_faces]


def check(detect_res):
    return min([len(faces) for faces in detect_res]) != 0


def detect_all(file, sfd_only=False, return_frames=False, max_size=None):
    frames = grab_all_frames(file, max_size=max_size, cvt=True)
    if not sfd_only:
        detect_res = flatten(
            [detector.detect(item) for item in partition(frames, 50)]
        )
        detect_res = get_valid_faces(detect_res, thres=0.5)
    else:
        raise NotImplementedError

    all_68 = get_lm68(frames, detect_res)
    if not return_frames:
        return detect_res, all_68
    else:
        return detect_res, all_68, frames


def get_lm68(frames, detect_res):
    assert len(frames) == len(detect_res)
    frame_count = len(frames)
    all_68 = []
    for i in range(frame_count):
        frame = frames[i]
        faces = detect_res[i]
        if len(faces) == 0:
            res_68 = []
        else:
            feeds = []
            for face in faces:
                assert len(face) == 3
                box = face[0]
                feed = LandmarkPredictor.prepare_feed(frame, box)
                feeds.append(feed)
            res_68 = predictor(feeds)
            assert len(res_68) == len(faces)
            for face, l_68 in zip(faces, res_68):
                if face[1] is None:
                    face[1] = get_five(l_68)
        all_68.append(res_68)

    assert len(all_68) == len(detect_res)
    return all_68

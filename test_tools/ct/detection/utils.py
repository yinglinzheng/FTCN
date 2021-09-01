import cv2
from test_tools.utils import flatten
import numpy as np


def chunks(l, n, step=None):
    if step is None:
        step = n
    return [l[i : i + n] for i in range(0, len(l), step)]


def sample_chunks(l, n, step=None):
    return [l[i : i + n] for i in range(0, len(l), step) if i + n <= len(l)]


def grab_all_frames(path, max_size, cvt=False):
    capture = cv2.VideoCapture(path)
    ret = True
    frames = []
    while ret:
        ret, frame = capture.read()
        if ret:
            if cvt:
                frame = frame[..., ::-1]
            frames.append(frame)
            if len(frames) == max_size:
                break
    capture.release()
    return frames


def get_clips_uniform(path, count, clip_size):
    capture = cv2.VideoCapture(path)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    max_clip_available = n_frames + 1 - clip_size
    if count > max_clip_available:
        count = max_clip_available
    final_start = max_clip_available - 1
    start_indices = np.linspace(0, final_start, count, endpoint=True, dtype=np.int)
    all_clip_idx = [list(range(start, start + clip_size)) for start in start_indices]
    valid = set(flatten(all_clip_idx))
    max_idx = max(valid)

    frames = {}
    for idx in range(max_idx + 1):
        # Get the next frame, but don't decode if we're not using it.
        ret = capture.grab()
        if not ret:
            continue

        if idx in valid:
            ret, frame = capture.retrieve()
            if not ret or frame is None:
                continue
            else:
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[idx] = frame

    capture.release()
    clips = []
    for clip_idx in all_clip_idx:
        clip = []
        flag = True
        for idx in clip_idx:
            if idx not in frames:
                flag = False
                break
            clip.append(frames[idx])
        if flag:
            clips.append(clip)
    return clips


def get_valid_faces(detect_results, max_count=10, thres=0.5, at_least=False):
    new_results = []
    for i, faces in enumerate(detect_results):
        if len(faces) > max_count:
            faces = faces[:max_count]
        l = []
        for j, face in enumerate(faces):
            if face[-1] < thres and not (j == 0 and at_least):
                continue
            box, lm, score = face
            box = box.astype(np.float)
            lm = lm.astype(np.float)
            l.append((box, lm, score))
        new_results.append(l)
    return new_results


def scale_box(box, scale_h, scale_w, h, w):
    x1, y1, x2, y2 = box.astype(np.int32)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    box_h = int((y2 - y1) * scale_h)
    box_w = int((x2 - x1) * scale_w)
    new_x1 = center_x - box_w // 2
    new_x2 = new_x1 + box_w
    new_y1 = center_y - box_h // 2
    new_y2 = new_y1 + box_h
    new_x1 = max(new_x1, 0)
    new_y1 = max(new_y1, 0)
    new_y2 = min(new_y2, h)
    new_x2 = min(new_x2, w)
    return new_x1, new_y1, new_x2, new_y2


def get_bbox(detect_res):
    tmp_detect_res = get_valid_faces(detect_res, max_count=4, thres=0.5)
    all_face_bboxs = []
    for faces in tmp_detect_res:
        all_face_bboxs.extend([face[0] for face in faces])
    all_face_bboxs = np.array(all_face_bboxs).astype(np.int)
    x1 = all_face_bboxs[:, 0].min()
    x2 = all_face_bboxs[:, 2].max()
    y1 = all_face_bboxs[:, 1].min()
    y2 = all_face_bboxs[:, 3].max()

    return x1, y1, x2, y2


def delta_detect_res(detect_res, x1, y1):
    diff = np.array([[x1, y1]])
    new_detect_res = []
    for faces in detect_res:
        f = []
        for face in faces:
            box, lm, score = face
            box = box.astype(np.float)
            box[[0, 2]] -= x1
            box[[1, 3]] -= y1
            lm = lm.astype(np.float) - diff
            f.append((box, lm, score))
        new_detect_res.append(f)
    return new_detect_res


def pre_crop(clips, detect_res):
    box = np.array(get_bbox(detect_res))
    w = box[2] - box[0]
    h = box[3] - box[1]
    x1, y1, x2, y2 = scale_box(
        box, 1.5, 1.2 if w > 2 * h else 1.5, clips[0].shape[0], clips[0].shape[1]
    )
    clips = np.array(clips)
    return clips[:, y1:y2, x1:x2], delta_detect_res(detect_res, x1, y1)

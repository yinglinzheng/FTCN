from .sort import Sort
import numpy as np


def get_detections(faces):
    detections = []
    for face in faces:
        x1, y1, x2, y2 = face[0]
        detections.append((x1, y1, x2, y2, face[-1]))
    return np.array(detections)


def get_tracks(detect_results):
    tracks = {}
    mot_tracker = Sort()
    for faces in detect_results:
        detections = get_detections(faces)
        track_bbs_ids = mot_tracker.update(detections)
        for track in track_bbs_ids:  # 单独框出每一张人脸
            id = int(track[-1])
            box = track[:4]
            if id in tracks:
                tracks[id].append(box)
            else:
                tracks[id] = [box]

    return [track for id, track in tracks.items() if len(track) == len(detect_results)]

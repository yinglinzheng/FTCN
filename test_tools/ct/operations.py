import os

import os
import cv2
import numpy as np
from .tracking.sort import iou


def face_iou(f1, f2):
    return iou(f1[0], f2[0])


def simple_tracking(batch_landmarks, index=0, thres=0.5):
    track = []

    for i, faces in enumerate(batch_landmarks):
        if i == 0:
            if len(faces) <= index or faces[index][-1] < 0.8:
                return None
            if index != 0:
                for idx in range(index):
                    if face_iou(faces[idx], faces[index]) > thres:
                        return None
            track.append(faces[index])
        else:
            last = track[i - 1]
            if len(faces) == 0:
                return None
            sorted_faces = sorted(faces, key=lambda x: face_iou(x, last), reverse=True)
            if face_iou(sorted_faces[0], last) < thres:
                return None
            track.append(sorted_faces[0])
    return track


def multiple_tracking(batch_landmarks):
    tracks = []
    for i in range(len(batch_landmarks[0])):
        track = simple_tracking(batch_landmarks, index=i)
        if track is None:
            continue
        tracks.append(track)
    return tracks

def find_longest(detect_res):
    fc = len(detect_res)
    tuples = []
    start = 0
    end = 0
    previous_count = -1
    all_tracks = []
    # start 取得到，end 取不到
    while start < (fc - 1):
        for end in range(start + 2, fc + 1):
            tracks = multiple_tracking(detect_res[start:end])
            if (len(tracks) != previous_count and previous_count != -1) or len(
                tracks
            ) == 0:
                break
            previous_count = len(tracks)
        if end - start > 2:
            if end != fc:
                un_reach_end = end - 1
            else:
                un_reach_end = end
            sub_tracks = multiple_tracking(detect_res[start:un_reach_end])
            if end == fc and len(sub_tracks) == 0:
                un_reach_end = end - 1
                sub_tracks = multiple_tracking(detect_res[start:un_reach_end])
            if len(sub_tracks) > 0:
                tpl = (start, un_reach_end)
                tuples.append(tpl)
                all_tracks.append(sub_tracks[0])
            else:
                raise NotImplementedError
            previous_count = -1
            end = un_reach_end
        start = end
    return tuples, all_tracks
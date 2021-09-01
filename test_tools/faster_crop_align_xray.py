import numpy as np
import cv2
from .warp_for_xray import (
    estimiate_batch_transform,
    transform_landmarks,
    std_points_256,
)
import numpy as np


class FasterCropAlignXRay:
    """
    修正到统一坐标系，统一图像大小到标准尺寸
    """

    def __init__(self, size=256):
        self.image_size = size
        self.std_points = std_points_256 * size / 256.0

    def __call__(self, landmarks, images=None, jitter=False):
        landmarks = [landmark[:4] for landmark in landmarks]
        ori_boxes = np.array([ori_box for _, _, _, ori_box in landmarks])
        five_landmarks = np.array([ldm5 for _, ldm5, _, _ in landmarks])
        landmarks68 = np.array([ldm68 for _, _, ldm68, _ in landmarks])
        # assert landmarks68.min() > 0

        left_top = ori_boxes[:, :2].min(0)

        right_bottom = ori_boxes[:, 2:].max(0)

        size = right_bottom - left_top

        w, h = size

        diff = ori_boxes[:, :2] - left_top[None, ...]

        new_five_landmarks = five_landmarks + diff[:, None, :]
        new_landmarks68 = landmarks68 + diff[:, None, :]

        landmark_for_estimiate = new_five_landmarks.copy()
        if jitter:
            landmark_for_estimiate += np.random.uniform(
                -4, 4, landmark_for_estimiate.shape
            )

        tfm, trans = estimiate_batch_transform(
            landmark_for_estimiate, tgt_pts=self.std_points
        )

        transformed_landmarks68 = np.array(
            [transform_landmarks(ldm68, trans) for ldm68 in new_landmarks68]
        )

        if images is not None:
            transformed_images = [
                self.process_sinlge(tfm, image, d, h, w)
                for image, d in zip(images, diff)
            ]  # 拼接 func 的参数
            transformed_images = np.stack(transformed_images)
            return transformed_landmarks68, transformed_images
        else:
            return transformed_landmarks68

    def process_sinlge(self, tfm, image, d, h, w):
        assert isinstance(image, np.ndarray)
        new_image = np.zeros((h, w, 3), dtype=np.uint8)
        x, y = d
        ih, iw, _ = image.shape
        new_image[y : y + ih, x : x + iw] = image
        transformed_image = cv2.warpAffine(
            new_image, tfm, (self.image_size, self.image_size)
        )
        return transformed_image

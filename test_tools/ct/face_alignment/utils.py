import cv2


def drawLandmark_multiple(img, bbox, landmark):
    """
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    """
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    return img

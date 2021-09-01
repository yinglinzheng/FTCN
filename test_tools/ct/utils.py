import cv2


def write_img(file, img):
    cv2.imwrite(file, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

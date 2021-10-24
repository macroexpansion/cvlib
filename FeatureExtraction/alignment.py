import cv2
import numpy as np

from .align import face_align

"""
    Face landmarks align
"""


def Alignment(image, landmark):
    if len(landmark) == 0:
        return None

    pts5 = np.array([landmark])
    pts5 = pts5[0, :]
    nimg = face_align.norm_crop(image, pts5)
    return nimg

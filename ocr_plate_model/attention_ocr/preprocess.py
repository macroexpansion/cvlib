import cv2
import numpy as np
import os

def resize_padding_oneline(image):
    h, w, _ = image.shape
    ratio = 64.0/h
    new_w = int(w*ratio)
    
    if new_w < 256:
        image = cv2.resize(image, (new_w, 64), interpolation=cv2.INTER_CUBIC)
        pad_img = np.ones((64, 256-new_w, 3), dtype=np.uint8)*127
        image = np.concatenate((image, pad_img), axis=1)
    else:
        image = cv2.resize(image, (256, 64), interpolation=cv2.INTER_CUBIC)
    return image
  
def resize_padding_twoline(image):
    h, w, _ = image.shape
    ratio = 128.0/h
    new_w = int(w*ratio)
    
    if new_w < 256:
        image = cv2.resize(image, (new_w, 128), interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(image, (256, 128), interpolation=cv2.INTER_CUBIC)
    return image

def preprocess(img, plate_shape):
    if plate_shape == 1:
        img = resize_padding_oneline(img)
        pad_img = np.ones((64, 256, 3), dtype=np.uint8)*127
        img = np.concatenate((img, pad_img), axis=0)
        # pad = (128-64)//2
        # img = np.pad(img, [(pad,), (0,)], mode='constant', constant_values=127)
    else:
        img = resize_padding_twoline(img)
        h, w, _ = img.shape
        pad = (256-w)//2
        img = np.pad(img, [(0,), (pad,), (0,)], mode='constant', constant_values=127)
        if (256 - w) % 2 == 1:
            pad_img = np.ones((h, 1, 3), dtype=np.uint8)*127
            img = np.concatenate((img, pad_img), axis=1)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


    
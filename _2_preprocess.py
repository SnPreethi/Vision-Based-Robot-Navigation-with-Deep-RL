# PREPROCESSING CAMERA IMAGES

# importing libraries
import cv2
import numpy as np

# the size of the target image after preprocessing
img_size = (84, 84)

def preprocess(img, size=img_size):
    # race conditions
    if img is None:
        raise ValueError("Received None image in preprocess()")

    # Image resize
    img = cv2.resize(img, size)

    # SANITY CHECK before transpose to catch camera misconfig
    assert img.ndim == 3 and img.shape[2] == 3, \
    f'Expected HxWx3 image, got shape {img.shape}'

    # Image normalization [0, 1]
    img = img.astype(np.float32) / 255.0

    # CHW format rearrangement
    img = np.transpose(img, (2, 0, 1))
    
    return img
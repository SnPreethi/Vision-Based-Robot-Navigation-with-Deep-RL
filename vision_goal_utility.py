# UTILITY FILE FOR VISIBILITY + HORIZONTAL OFFSET OF GOAL
'''
Detect the goal visually. Conver image to HSV, threshold for purple, find pixels belonging to goal and decide visibility + horizontal offset
'''

# importing libraries
import cv2
import numpy as np

# detect goal marker in an RGB image
def detect_goal_marker(rgb_img):
    # colour detection of goal RGB -> HSV
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

    # defining purple colour range
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])

    # threshold
    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # check visibility
    pixel_count = np.count_nonzero(mask)
    if pixel_count < 50:
        return False, 0.0

    # computing horizontal offset
    h, w = mask.shape
    xs = np.where(mask > 0)[1]
    
    if xs.size == 0:
        return False, 0.0

    x_center = np.mean(xs)
    image_center = w / 2.0

    goal_x_offset = (x_center - image_center) / image_center
    goal_x_offset = float(np.clip(goal_x_offset, -1.0, 1.0))

    return True, goal_x_offset
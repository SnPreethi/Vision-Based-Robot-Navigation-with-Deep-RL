# Create test_vision.py
import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/administrator/RL_Pipeline/2_RL_/1_Vision_Only/1_Deep_RL')
from vision_goal_utility import detect_goal_marker

# Create a synthetic purple image to test detection
test_img = np.zeros((480, 640, 3), dtype=np.uint8)

# Draw a purple rectangle in center (RGB: 128, 0, 128)
test_img[200:280, 280:360] = [128, 0, 128]

visible, offset = detect_goal_marker(test_img)
print(f"Synthetic purple detection: visible={visible}, offset={offset}")
# Should print: visible=True, offset=0.0 (centered)

# Test with slightly off-center
test_img2 = np.zeros((480, 640, 3), dtype=np.uint8)
test_img2[200:280, 400:480] = [128, 0, 128]  # Right side

visible2, offset2 = detect_goal_marker(test_img2)
print(f"Right-side detection: visible={visible2}, offset={offset2}")
# Should print: visible=True, offset~=+0.5 (right of center)
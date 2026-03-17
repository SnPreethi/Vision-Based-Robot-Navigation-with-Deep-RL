# TEMPORAL FRAME STACKING FOR MOTION CUE

# importing libraries
import numpy as np
from collections import deque

class FrameStack:
    def __init__(self, k):
        # number of frames to stack (temporal window size)
        self.k = k
        self.frames = deque(maxlen=k)

    # reset frame history at the start of new episode
    def reset(self):
        self.frames.clear()

    # Adding new frame to the buffer
    def push(self, frame):
        assert isinstance(frame, np.ndarray), 'Frame must be numpy array'
        assert frame.ndim == 3, f'Expected (C, H, W), got {frame.shape}'
        self.frames.append(frame)

    # Returning stacked frames as a single tensor
    def get(self):
        assert len(self.frames) == self.k, 'FrameStack not full'
        return np.concatenate(self.frames, axis=0)  # (k*C, H, W)

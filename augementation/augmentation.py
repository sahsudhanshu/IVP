"""Clip-consistent photometric augmentation for BGR uint8 video frames."""
import numpy as np

def sample_transform(rng):
    """Draw once per clip, preserving temporal consistency."""
    if rng.random() >= 0.5:
        return (1.0, 0.0)
    return (rng.uniform(0.9, 1.1), rng.uniform(-8.0, 8.0))

def apply_transform(frame, params):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected H x W x 3 uint8 BGR frame")
    contrast, brightness = params
    return np.clip(frame.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

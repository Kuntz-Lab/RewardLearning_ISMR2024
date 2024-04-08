import os
import sys
import numpy as np
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

try:
    import cv2
    from cv_bridge import CvBridge
except ModuleNotFoundError:
    pass

from ll4ma_util import ui_util


# Utility functions for working with data (scaling, data representations
# (e.g. rotations), error checking on dataset types, etc.


def scale_min_max(x, x_min, x_max, desired_min, desired_max):
    """
    Applies min/max scaling on one data instance.

    Args:
        x (ndarray): Data to be scaled.
        x_min (flt): Minimum value of data (over full dataset).
        x_max (flt): Maximum value of data (over full dataset).
        desired_min (flt): Desired minimum value.
        desired_max (flt): Desired maximum value.
    """
    return ((desired_max - desired_min) * (x - x_min) / (x_max - x_min)) + desired_min
    

def segmentation_to_rgb(segmentation, seg_ids, colors):
    """
    Converts single-channel segmentation mask (each unique value corresponds to one mask)
    to an RGB image where each class gets its own color.

    Args:
        segmentation (ndarray): Single channel array of segmentation values.
        seg_ids (list): The unique ids contained in the image.
        colors (list): List of RGB color values (tuples)
    Returns:
        rgb (ndarray): 3-channel RGB image array
    """
    height, width = segmentation.shape
    rgb = np.zeros((height, width, 3))
    for seg_id, color in zip(seg_ids, colors):
        rgb[segmentation == seg_id] = color
    rgb *= 255
    rgb = rgb.astype(np.uint8)
    return rgb


def segmentation_to_k_channel(segmentation, seg_ids):
    """
    Converts single-channel segmentation mask (each unique value corresponds to one mask)
    Args:
        segmentation (ndarray): Single channel array of segmentation values.
        seg_ids (list): The unique ids contained in the image.
    Returns:
        k_channel (ndarray): k-channel array, one channel for each segmentation mask.
    """
    if segmentation.ndim == 2:
        k_channel = segmentation == seg_ids[:, None, None]
    elif segmentation.ndim == 3:
        segmenation = np.transpose(segmentation, (1, 0, 2))
        seg_ids = seg_ids
        k_channel = segmentation == seg_ids[:, None, None, None]
        k_channel = np.transpose(k_channel, (1, 0, 2, 3))
    else:
        raise ValueError(f"Cannot handle shape {segmentation.shape}")
    return k_channel


def k_channel_to_rgb(k_channel, seg_ids, colors):
    if k_channel.ndim != 3:
        raise ValueError(f"Expected k_channel to have 3 dims but got shape {k_channel.shape}")
    single_channel = np.zeros(k_channel.shape[1:])
    for i, seg_id in enumerate(seg_ids):
        single_channel += seg_id * k_channel[i]
    rgb = segmentation_to_rgb(single_channel, seg_ids, colors)
    return rgb


def resize_images(imgs, shape):
    """
    Assuming imgs is over time (i.e. shape (t, h, w, c) or (t, h, w)), and shape
    is desired shape of either (h, w, c) or (h, w).
    """
    resized = np.zeros((len(imgs), *shape), dtype=imgs.dtype)
    for i in range(len(imgs)):
        resized[i] = cv2.resize(imgs[i], shape[:2], interpolation=cv2.INTER_NEAREST)
    return resized

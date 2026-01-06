# File containing functions that are shared between both files (steganography_embedding.py and steganography_extraction.py)

import cv2
import numpy as np
import pywt
from typing import Any


def load_image(img_path: str) -> np.ndarray:
    """
    Load an image from a file path and return a RGB numpy array.

    :param img_path: str
        Path to the image file.
    :return: np.ndarray
        RGB numpy array.
    """

    img = cv2.imread(img_path)
    return img


def convert_image(img: np.ndarray, color_space: str) -> np.ndarray:
    """
        Convert an image to a specified color space.

    :param img: np.ndarray
        Input image as a numpy array.
    :param color_space: str
        Target color space. Supported values are:
        "RGB", "GRAYSCALE", "RGB2YCrCb", "YCrCb2BGR".
    :return: np.ndarray
        Image converted to the specified color space as a numpy array.
    """

    conversions = {
        "RGB": cv2.COLOR_BGR2RGB,
        "GRAYSCALE": cv2.COLOR_BGR2GRAY,
        "RGB2YCrCb": cv2.COLOR_RGB2YCrCb,  # In cv2 the order is YCrCb and not the standard YCbCr
        "YCrCb2BGR": cv2.COLOR_YCrCb2BGR
    }

    if color_space in conversions:
        converted_img = cv2.cvtColor(img, conversions[color_space])
    else:
        raise ValueError(f"Unknown color space: {color_space}")

    return converted_img


def resize_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """
        Resize an image to the specified width and height.

    :param img: np.ndarray
        Input image as a numpy array.
    :param width: int
        Target width of the resized image.
    :param height: int
        Target height of the resized image.
    :return: np.ndarray
        Resized image as a numpy array.
    """

    resized_img = cv2.resize(
        img,
        (width, height)
    )

    return resized_img


def split_ycrcb_channels(ycrcb_img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Split a YCrCb image into its individual channels.

    :param ycrcb_img: np.ndarray
        Image in YCrCb color space.
    :return: tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - Y (luminance) channel
        - Cr (chrominance-red) channel
        - Cb (chrominance-blue) channel
    """

    y_channel, cr_channel, cb_channel = cv2.split(ycrcb_img)
    return y_channel, cr_channel, cb_channel


def discrete_wavelet_transform(channel: np.ndarray, wavelet: str, mode: str, level: int) -> list[Any]:
    """
        Apply a 2D discrete wavelet transform (DWT) to an image channel.

    :param channel: np.ndarray
        Single image channel (Cb channel) as a numpy array.
    :param wavelet: str
        Name of the wavelet to use (e.g. "haar", "db1", "db2").
    :param mode: str
        Signal extension mode used for the wavelet transform
        (e.g. "symmetric", "periodization").
    :param level: int
        Decomposition level of the wavelet transform.
    :return: list[Any]
        Wavelet decomposition coefficients in the format returned by
        pywt.wavedec2:
        [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
    """

    coefficients = pywt.wavedec2(channel, wavelet=wavelet, mode=mode, level=level)
    return coefficients

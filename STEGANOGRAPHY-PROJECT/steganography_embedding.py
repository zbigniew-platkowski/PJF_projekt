# File responsible for the embedding process of the secret image into cover.

import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
#1 import streamlit as st
from typing import Any
from dataclasses import dataclass
from skimage.metrics import \
    peak_signal_noise_ratio as psnr  # PSNR - To assess the imperceptibility (niedostrzegalność) of the secret image,
                                     # PSNR is used to calculate the stego image quality

import common
from config_embedding import PARAMS


@dataclass
class CoverProperties:
    """
        Container for cover image properties used during the embedding process.

    :param original_image: np.ndarray
        Original cover image in BGR color space.
    :param y_channel: np.ndarray
        Luminance (Y) channel of the cover image.
    :param cr_channel: np.ndarray
        Chrominance-red (Cr) channel of the cover image.
    :param cb_channel: np.ndarray
        Chrominance-blue (Cb) channel of the cover image,
        used as the host channel for data embedding.
    """

    original_image: np.ndarray
    y_channel: np.ndarray
    cr_channel: np.ndarray
    cb_channel: np.ndarray


@dataclass
class SecretProperties:
    """
        Container for secret image properties used during the embedding process.

    :param original_image: np.ndarray
        Original secret image.
    :param grayscale_image: np.ndarray
        Grayscale version of the secret image used for embedding.
    """

    original_image: np.ndarray
    grayscale_image: np.ndarray


@dataclass
class StegoProperties:
    """
        Container for stego image properties and extraction parameters.

    :param image: np.ndarray
        Stego image containing the embedded secret.
    :param psnr: float
        Peak Signal-to-Noise Ratio (PSNR) measuring stego image quality.
    :param secret_key: dict
        Dictionary containing parameters required for the extraction process.
    """

    image: np.ndarray
    psnr: float
    secret_key: dict


@dataclass
class EmbedResult:
    """
        Container wrapping all results of the embedding process.

    :param cover: CoverProperties
        Properties of the cover image.
    :param secret: SecretProperties
        Properties of the secret image.
    :param stego: StegoProperties
        Properties of the stego image, including quality metrics
        and the secret key required for extraction.
    """

    cover: CoverProperties
    secret: SecretProperties
    stego: StegoProperties


def add_to_secret_key(secret_key: dict[str, Any], **kwargs: Any) -> None:
    """
    Updates the secret key (symmetric cryptography) in-place with parameters needed for the extraction process.

    :param secret_key: dict
        Dictionary storing parameters required for extraction.
    :param kwargs:
        Arguments to be added to the secret key.
    :return: None
        The dictionary is modified in place (mutable object).
    """

    secret_key.update(kwargs)


def show_deepest_decomposition(coefficients: list[np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]], dwt_level: int) -> None:
    """
    Visualize the deepest level of the 2D discrete wavelet decomposition.

    :param coefficients: list
        Wavelet decomposition coefficients returned by pywt.wavedec2.
    :param dwt_level: int
        Decomposition level to visualize.
    :return: None
        Displays the decomposition using matplotlib and Streamlit.
    """

    cA = coefficients[0]
    (cH, cV, cD) = coefficients[-dwt_level]  # [-2] means second decomposition level, similarly [1] (but that's less intuitive)
                                             # [-1] means first decomposition level, similarly [2]

    titles = [
        'Approximation Detail (LL) cA',
        'Horizontal Detail (LH) cH',
        'Vertical Detail (HL) cV',
        'Diagonal Detail (HH) cD'
    ]

    fig = plt.figure(figsize=(8, 8))

    for i, c in enumerate([cA, cH, cV, cD]):
        plt.subplot(2, 2, i + 1)

        cmap = 'gray' if i == 0 else 'seismic'
        plt.imshow(c, cmap=cmap)

        plt.title(f"{titles[i]}{dwt_level}", fontsize=10)
        plt.colorbar() if i != 0 else None

    plt.tight_layout()
    #1 st.pyplot(fig)
    plt.close(fig)


def embed_secret_into_coefficient_svd(host_coefficient: np.ndarray, secret_img: np.ndarray, alpha: float, secret_key: dict[str, Any]) -> np.ndarray:
    """
    Embed a secret image into a host DWT coefficient using SVD.

    :param host_coefficient: np.ndarray
        DWT coefficient of the cover image to be modified.
    :param secret_img: np.ndarray
        Grayscale secret image resized to match the host coefficient.
    :param alpha: float
        Embedding strength parameter.
    :param secret_key: dict
        Dictionary used to store data required for extraction.
    :return: np.ndarray
        Modified DWT coefficient containing the embedded secret.
    """

    U_host, S_host, V_host = np.linalg.svd(host_coefficient, full_matrices=False)  # False means economic SVD
    U_secret, S_secret, V_secret = np.linalg.svd(secret_img, full_matrices=False)

    add_to_secret_key(
        secret_key,
        S_host=S_host,
        U_secret=U_secret,
        V_secret=V_secret
    )

    S_host_mod = np.diag(S_host + alpha * S_secret)

    return U_host @ S_host_mod @ V_host


def merge_ycrcb_channels(y_channel: np.ndarray, cr_channel: np.ndarray, cb_channel: np.ndarray) -> np.ndarray:
    """
    Merge Y, Cr, and Cb channels into a single YCrCb image.

    :param y_channel: np.ndarray
        Luminance channel
    :param cr_channel: np.ndarray
        Chrominance-red channel.
    :param cb_channel: np.ndarray
        Chrominance-blue channel.
    :return: np.ndarray
        Merged YCrCb image.
    """

    ycrcb_img = cv2.merge([y_channel, cr_channel, cb_channel])
    return ycrcb_img


def inverse_discrete_wavelet_transform(coefficients_mod: list[np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]], wavelet: str, mode: str) -> np.ndarray:
    """
    Perform inverse 2D discrete wavelet transform.

    :param coefficients_mod: list
        Modified wavelet coefficients.
    :param wavelet: str
        Wavelet type used for reconstruction.
    :param mode: str
        Inverse discrete wavelet transform mode.
    :return: np.ndarray
        Reconstructed image channel.
    """

    return pywt.waverec2(coefficients_mod, wavelet=wavelet, mode=mode)


def embed(cover_img: np.ndarray, secret_img: np.ndarray) -> EmbedResult:
    """
    Embed a secret image into a cover image using DWT-SVD steganography.

    :param cover_img: np.ndarray
        Original cover image in BGR color space.
    :param secret_img: np.ndarray
        Original secret image.
    :return: EmbedResult
        Object containing cover properties, secret properties,
        stego image, PSNR value, and secret key.
    """

    # Preserve original inputs
    cover_original = cover_img.copy()
    secret_original = secret_img.copy()

    # Convert cover image to YCrCb color space
    cover_rgb = common.convert_image(cover_img, "RGB")
    cover_ycrcb = common.convert_image(cover_rgb, "RGB2YCrCb")

    # Convert secret image to grayscale
    secret_grayscale = common.convert_image(secret_img, "GRAYSCALE")

    # Split cover image into Y, Cr and Cb channels
    y_channel, cr_channel, cb_channel = common.split_ycrcb_channels(cover_ycrcb)

    # Select Cb channel for data embedding
    channel_to_mod = cb_channel

    # Perform DWT on the selected channel and store the result in coefficients channel
    coeffs_channel = common.discrete_wavelet_transform(
        channel_to_mod,
        PARAMS["dwt_wavelet"],
        PARAMS["dwt_mode"],
        PARAMS["dwt_level"]
    )

    # Initialize secret key dictionary
    secret_key = {}

    # Store embedding parameters required for extraction
    add_to_secret_key(
        secret_key,
        dwt_wavelet=PARAMS["dwt_wavelet"],
        dwt_mode=PARAMS["dwt_mode"],
        dwt_level=PARAMS["dwt_level"],
        alpha=PARAMS["alpha"],
        secret_width=secret_original.shape[1],
        secret_height=secret_original.shape[0]
    )

    # Visualize the deepest DWT decomposition level
    show_deepest_decomposition(coeffs_channel, PARAMS["dwt_level"])

    cA2 = coeffs_channel[0]  # LL 2nd level
    (cH2, cV2, cD2) = coeffs_channel[-2]  # LH, HL, HH 2nd level
    (cH1, cV1, cD1) = coeffs_channel[-1]  # LH, HL, HH 1st level

    # Select coefficient of the previously chosen channel to be modified
    coeff_channel_to_mod = cV2

    # Resize secret image to match the selected DWT coefficient size
    secret = common.resize_image(
        secret_grayscale,
        coeff_channel_to_mod.shape[1], coeff_channel_to_mod.shape[0]
    )

    # secret = secret.astype(np.float32) ?

    # Embed the secret image into the selected DWT coefficient using SVD
    coeff_channel_mod = embed_secret_into_coefficient_svd(
        coeff_channel_to_mod,
        secret,
        PARAMS["alpha"],
        secret_key
    )

    # Reconstruct modified wavelet coefficients
    coeffs_channel_mod = [
        cA2,
        (cH2, coeff_channel_mod, cD2),
        (cH1, cV1, cD1)
    ]

    # Apply inverse DWT to reconstruct the modified channel
    channel_mod = inverse_discrete_wavelet_transform(
        coeffs_channel_mod,
        PARAMS["dwt_wavelet"],
        PARAMS["dwt_mode"]
    )

    # Normalize pixel values and convert to uint8 for OpenCV compatibility
    channel_mod = np.clip(np.rint(channel_mod), 0, 255).astype(np.uint8)

    # 1. rint - rounds the result of SVD/DWT to the nearest integer using banker's rounding
    # 2. clip - makes sure, the pixels are in the range of [0, 255]
    # 3. astype(uint8) - prepares for OpenCV (unsigned integer)

    # Merge modified channel back with original Y and Cr channels
    stego_ycrcb = merge_ycrcb_channels(y_channel, cr_channel, channel_mod)

    # Convert stego image back to BGR color space
    stego_bgr = common.convert_image(stego_ycrcb, "YCrCb2BGR")

    # Wrap properties of the cover image used during the embedding process
    cover_props = CoverProperties(
        original_image=cover_original,
        y_channel=y_channel,
        cr_channel=cr_channel,
        cb_channel=cb_channel
    )

    # Wrap properties of the secret image used during the embedding process
    secret_props = SecretProperties(
        original_image=secret_original,
        grayscale_image=secret_grayscale,
    )

    # Wrap properties of the stego image and parameters required for extraction
    stego_props = StegoProperties(
        image=stego_bgr,
        psnr=psnr(cover_original, stego_bgr, data_range=255),
        secret_key=secret_key
    )

    # Wrap all embedding-related outputs into a single EmbedResult dataclass
    embed_result = EmbedResult(
        cover=cover_props,
        secret=secret_props,
        stego=stego_props
    )

    return embed_result

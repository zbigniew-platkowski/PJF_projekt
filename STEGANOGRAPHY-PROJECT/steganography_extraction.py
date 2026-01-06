# File responsible for the extracting process of a secret image from a stego image.

import numpy as np
from typing import Any
from dataclasses import dataclass
from skimage.metrics import structural_similarity as ssim  # SSIM - Used to calculate the degree of similarity between
                                                           # the extracted secret image and the original secret image

import common


@dataclass
class ExtractResult:
    """
        Container for extraction results.

    :param image: np.ndarray
        Extracted secret image.
    :param ssim: float
        Structural Similarity Index (SSIM) between the extracted
        and the original secret image.
    """

    image: np.ndarray
    ssim: float


def extract(stego_img: np.ndarray, original_grayscale_secret_img: np.ndarray, secret_key: dict[str, Any]) -> ExtractResult:
    """
        Extract a hidden secret image from a stego image.

        This function performs the extraction stage of the steganographic
        process using DWT and SVD. The secret image is recovered from the
        chrominance channel (Cb) of the stego image and compared against
        the original secret image using SSIM.

    :param stego_img: np.ndarray
        Stego image containing the hidden secret.
    :param original_grayscale_secret_img: np.ndarray
        Original secret image in grayscale, used only for SSIM evaluation.
    :param secret_key:
        Dictionary containing extraction parameters and keys, including (in order of being added):
        - dwt_wavelet
        - dwt_mode
        - dwt_level
        - alpha
        - secret_width
        - secret_height
        - S_host
        - U_secret
        - V_secret
    :return: ExtractResult
        Dataclass containing the extracted secret image and its SSIM score.
    """

    # Convert stego image to YCrCb color space
    stego_rgb = common.convert_image(stego_img, "RGB")
    stego_ycrcb = common.convert_image(stego_rgb, "RGB2YCrCb")

    # Ensure original secret image is single-channel grayscale
    secret_grayscale = common.convert_image(original_grayscale_secret_img,"GRAYSCALE")
    # OpenCV loads images as 3-channel by default (even if it's already grayscale).

    secret_grayscale_f= secret_grayscale.astype(np.float32)


    # Extract Cb channel (used for hiding data)
    _, _, cb_channel = common.split_ycrcb_channels(stego_ycrcb)
    channel_with_secret = cb_channel  # Channel with hidden data

    # Apply DWT on the selected channel
    coeffs_channel = common.discrete_wavelet_transform(
        channel_with_secret,
        str(secret_key["dwt_wavelet"]),
        str(secret_key["dwt_mode"]),
        secret_key["dwt_level"]
    )

    # Extract high-frequency vertical coefficients at selected level
    (cH2, cV2, cD2) = coeffs_channel[-secret_key["dwt_level"]]
    coeff_channel_with_secret = cV2

    # Singular Value Decomposition
    _, S_host_with_secret, _ = np.linalg.svd(coeff_channel_with_secret, full_matrices=False)

    # Recover singular values of the secret image
    S_secret_extracted = (S_host_with_secret - secret_key["S_host"]) / float(secret_key["alpha"])  # Order different from the one in article ?
    S_secret_extracted = np.diag(S_secret_extracted)

    # Reconstruct the secret image
    secret_recovered_f = (secret_key["U_secret"] @ S_secret_extracted @ secret_key["V_secret"]).astype(np.float32)

    # Resize she recovered secret image to its original dimensions
    secret_recovered_f = common.resize_image(
        secret_recovered_f,
        int(secret_key["secret_width"]), int(secret_key["secret_height"])
    )

    # Clip values and convert to uint8
    secret_recovered = np.clip(np.rint(secret_recovered_f), 0, 255).astype(np.uint8)

    # Wrap the extracted secret image and its SSIM score into and ExtractResult dataclas
    extract_result = ExtractResult(
        image=secret_recovered,
        ssim=ssim(secret_recovered_f, secret_grayscale_f, data_range=255) # Compute SSIM score
    )

    return extract_result

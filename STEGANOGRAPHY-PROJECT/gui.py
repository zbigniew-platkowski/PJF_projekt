# File responsible for rendering the gui using streamlit.

import streamlit as st
import numpy as np
import cv2
import io
from typing import Any
from steganography_embedding import embed
from steganography_extraction import extract
from streamlit.runtime.uploaded_file_manager import UploadedFile


def uploaded_img_to_np(uploaded_file: UploadedFile) -> np.ndarray:
    """
    Decodes an uploaded image file into a NumPy array using OpenCV.

    :param uploaded_file: UploadedFile
        Image uploaded via Streamlit.
    :return:
        Image decoded in BGR color space.
    """

    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)

    # Decode uploaded image bytes into a BGR OpenCV image
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def uploaded_npz_to_dict(uploaded_file: UploadedFile) -> dict[str, Any]:
    """
    Loads a NumPy .npz file uploaded via Streamlit into a dictionary.

    :param uploaded_file: UploadedFile
        Numpy .npz file uploaded via Streamlit.
    :return:
        Dictionary storing parameters required for extraction.
    """

    # Load .npz file from in-memory buffer
    buf = io.BytesIO(uploaded_file.getvalue())  # Buf isn't necessary ?
    data = np.load(buf, allow_pickle=True)
    return dict(data)


def upload_and_preview_img(label: str, key: str, file_types: tuple[str, ...] =("png",)) -> UploadedFile:
    """
    Upload an image file and display a preview in the UI.

    :param label: str
        Label of the image.
    :param key: str
        Key of the image.
    :param file_types: tuple[str, ...]
        Allowed file extensions.
    :return: UploadedFile
        UploadedFile object.
    """

    # Upload image file via Streamlit
    img = st.file_uploader(
        label,
        type=list(file_types),
        key=key
    )

    # Display preview if an image was uploaded
    if img:
        st.image(
            img,
            caption=label,
            use_container_width=True
        )

    return img


def render_gui() -> None:
    """
    Render the main GUI and route between Embed and Extract modes.

    :return: None
    """

    st.title("Image steganography")

    mode = st.radio(
        "Choose whether to embed or extract the secret image:",
        ("Embed", "Extract"),
        horizontal=True,
        key="mode"
    )

    if "last_mode" not in st.session_state:
        st.session_state["last_mode"] = st.session_state["mode"]

    if st.session_state["mode"] != st.session_state["last_mode"]:
        # Reset embed/extract results on mode switch
        st.session_state.pop("embed_result", None)
        st.session_state.pop("extract_result", None)

        st.session_state["last_mode"] = st.session_state["mode"]

    if mode == "Embed":
        render_embed_ui()
    else:
        render_extract_ui()


def render_embed_ui() -> None:
    """
    Render the main Embed UI.

    :return: None
    """

    col1, _, col3, _, col5 = st.columns([1, 0.5, 1, 0.5, 1])

    # Upload and preview cover image
    with col1:
        cover_file = upload_and_preview_img("Cover image", "cover")

    # Upload and preview secret image
    with col5:
        secret_file = upload_and_preview_img("Secret image", "secret")

    with col3:
        if cover_file and secret_file:
            # Convert uploaded images to NumPy arrays
            cover = uploaded_img_to_np(cover_file)
            secret = uploaded_img_to_np(secret_file)

            st.success("Both images successfully loaded")

            if st.button("Embed secret into cover", use_container_width=True):
                # Perform embedding process and save session state
                st.session_state["embed_result"] = embed(cover, secret)

            if "embed_result" in st.session_state:
                embed_result = st.session_state["embed_result"]

                col1, col2, col3 = st.columns(3)

                # Display Y, Cr and Cb channels of the cover image
                with col1:
                    st.image(embed_result.cover.y_channel, caption="Cover Y channel", use_container_width=True)

                with col2:
                    st.image(embed_result.cover.cr_channel, caption="Cover Cr channel", use_container_width=True)

                with col3:
                    st.image(embed_result.cover.cb_channel, caption="Cover Cb channel", use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.image(embed_result.stego.image, channels="BGR",
                             caption=f"Stego image | PSNR: {embed_result.stego.psnr:.2f}", use_container_width=True)

                    encode_success, buffer = cv2.imencode(".png", embed_result.stego.image)

                    st.download_button(
                        "Download stego (cover + secret) image",
                        data=buffer.tobytes(),
                        file_name="stego.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with col2:
                    st.image(embed_result.secret.grayscale_image, caption="Secret grayscale", use_container_width=True)

                    encode_success, buffer = cv2.imencode(".png", embed_result.secret.grayscale_image)

                    st.download_button(
                        "Download secret grayscale image",
                        data=buffer.tobytes(),
                        file_name="secret_grayscale.png",
                        mime="image/png",
                        use_container_width=True
                    )

                # Save secret key parameters to a .npz file for later extraction
                buf = io.BytesIO()
                np.savez(buf, **embed_result.stego.secret_key)
                buf.seek(0)

                st.download_button(
                    label="Download secret key (needed for extraction)",
                    data=buf,
                    file_name="secret_key.npz",
                    mime="application/octet-stream",
                    use_container_width=True
                )

        else:
            st.info("Load both images")


def render_extract_ui() -> None:
    """
    Render the main Extract UI.

    :return: None
    """

    col1, _, col3, _, col5 = st.columns([1, 0.5, 1, 0.5, 1])

    # Upload stego image
    with col1:
        stego_file = upload_and_preview_img("Stego image", "stego")

    # Upload secret key file
    with col5:
        secret_key_file = st.file_uploader(
            "Secret key",
            type=["npz"],
            key="secret_key"
        )

    # Upload original secret image for SSIM comparison
    with col3:
        original_secret_file = st.file_uploader(
            "Original grayscale secret image",
            type=["png"],
            key="original_secret"
        )

        if stego_file and secret_key_file and original_secret_file:
            # Convert uploaded images to NumPy arrays
            stego = uploaded_img_to_np(stego_file)
            original_grayscale_secret = uploaded_img_to_np(original_secret_file)

            # Load secret key parameters stored in a NumPy .npz file
            secret_key = uploaded_npz_to_dict(secret_key_file)

            st.success("All files successfully loaded")

            if st.button("Extract secret from stego", use_container_width=True):
                # Perform extraction process and save session state
                st.session_state["extract_result"] = extract(stego, original_grayscale_secret, secret_key)

            if "extract_result" in st.session_state:
                extract_result = st.session_state["extract_result"]

                col1, col2 = st.columns(2)

                with col1:
                    st.image(original_grayscale_secret, caption="Original grayscale secret image",
                             use_container_width=True)

                with col2:
                    st.image(extract_result.image, caption=f"Recovered secret image",
                             use_container_width=True)
                    # st.image(extract_result.image, caption=f"Recovered secret image | SSIM: {extract_result.ssim:.2f}",
                    #          use_container_width=True) # SSIM is calculated but the result is completely wrong when it comes to some images, not sure why.
                                                         # That it why it's commented out.

                encode_success, buffer = cv2.imencode(".png", extract_result.image)
                st.download_button(
                    "Download recovered secret image",
                    data=buffer.tobytes(),
                    file_name="recovered_secret.png",
                    mime="image/png",
                    use_container_width=True
                )

        else:
            st.info("Load all files")

This version is the finished project (organized, with GUI and everything described in detail).

The program uses discrete wavelet transform (DWT) and singular value decomposition (SVD) in both the embedding and extraction processes.
The secret image is embedded in the Cb channel of the cover image.

The implementation is based on the algorithm in the article available at
https://www.researchgate.net/publication/344533885_Image_Steganography_Based_on_Wavelet_Transform_and_Color_Space_Approach
where the detailed embedding and extraction procedures are described.

Input images in the embedding process should be color images (RGB) with the .png extension.

Launching the app
In PyCharm in terminal you need to navigate to the directory containing all the files. After that execute: streamlit run app.py.

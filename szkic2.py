import cv2
import pywt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def load_images(cover_img_path, secret_img_path):
    cover_img = cv2.imread(cover_img_path)
    cover_img = cv2.cvtColor(cover_img, cv2.COLOR_BGR2RGB) # Konwersja z BGR do RGB,
                                                           # bo domyślnie cv2 wczytuje BGR

    secret_img = cv2.imread(secret_img_path)
    secret_img = cv2.cvtColor(secret_img, cv2.COLOR_BGR2GRAY)

    return cover_img, secret_img

cover, secret = load_images("img/og_img.png", "img/secret_img.bmp")

def split_ycrcb(rgb_img):
    ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
    Y_channel, Cr_channel, Cb_channel = cv2.split(ycrcb_img)

    cv2.imwrite("img/Y.png", Y_channel)
    cv2.imwrite("img/Cr.png", Cr_channel)
    cv2.imwrite("img/Cb.png", Cb_channel)

    return Y_channel, Cr_channel, Cb_channel

Y, Cr, Cb = split_ycrcb(cover)
# 
# img1 = cv2.merge([Y, Cr, Cb])
# img1 = cv2.cvtColor(img1, cv2.COLOR_YCrCb2RGB)
# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
#
# cv2.imwrite("img/og_img_merged.png", img1)
# bo opencv działa natywnie w bgr, imwrite zakłada bgr, wiec trzeba zmienic z rgb na bgr




plt.figure(figsize=(16, 5))

plt.subplot(1, 5, 1)
plt.imshow(cover)
plt.title("Obraz oryginalny")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(cover)
plt.title("Obraz w przestrzeni barw YCbCr")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(Y, cmap='gray')
plt.title("Składowa Y")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(Cb, cmap='gray')
plt.title("Składowa Cb")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(Cr, cmap='gray')
plt.title("Składowa Cr")
plt.axis("off")

plt.show()

mother_wavelet = 'db1'
dwt_level = 2 # Liczba poziomów dekompozycji w dyskretnej transformacji falkowej (DWT)

coefficients = pywt.wavedec2(Cb, wavelet=mother_wavelet, mode='symmetric', level=dwt_level)

cA2 = coefficients[0] # LL poziom drugi
(cH2, cV2, cD2) = coefficients[-2] # LH, HL, HH poziom drugi
(cH1, cV1, cD1) = coefficients[-1]

plt.subplot(2, 2, 1)
plt.imshow(cA2, cmap='gray')
plt.title('cA2', fontsize=30)
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(cH2, cmap='seismic')
plt.title('cH2', fontsize=30)
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(cV2, cmap='seismic')
plt.title('cV2', fontsize=30)
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(cD2, cmap='seismic')
plt.title('cD2', fontsize=30)
plt.colorbar()

plt.show()

secret_img_gray_resized = cv2.resize(secret, (cV2.shape[1], cV2.shape[0])) # Zmiana rozmiaru sekretnego obrazka na identyczny rozmiar obrazu og
# shape[0] - liczba wierszy (wysokość obrazu)
# shape[1] - liczba kolumn (szerokość obrazu)

u_og_img, s_og_img, v_og_img = np.linalg.svd(cV2)
sigma_og = np.diag(s_og_img)

u_secret_img, s_secret_img, v_secret_img = np.linalg.svd(secret_img_gray_resized)
sigma_secret = np.diag(s_secret_img)

s_og_img_mod = sigma_og + 0.05 * sigma_secret

print(sigma_og)

print(sigma_secret)

print("\n",s_og_img_mod)

cV2_modified = u_og_img @ s_og_img_mod @ v_og_img

cA1_modified = pywt.idwt2((cA2, (cH2, cV2_modified, cD2)), wavelet=mother_wavelet, mode='symmetric')

Cb_modified = pywt.idwt2((cA1_modified, (cH1, cV1, cD1)), wavelet=mother_wavelet, mode='symmetric')

Cb_modified_u8 = np.clip(np.rint(Cb_modified), 0, 255).astype(np.uint8)

cv2.imwrite("img/1.png", Cb)
cv2.imwrite("img/2.png", Cb_modified_u8)


plt.subplot(1, 2, 1)
plt.imshow(Cb, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(Cb_modified, cmap='gray')
plt.show()

img1 = cv2.merge([Y, Cr, Cb_modified_u8])
img1 = cv2.cvtColor(img1, cv2.COLOR_YCrCb2RGB)
img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)

cv2.imwrite("img/og_img_reconstructed.png", img1)


# DWT

# plt.figure(figsize=(16, 12))
#
# plot_idx = 1   # licznik subplotów
#
# for current_level in range(1, dwt_level+1):
#     LL, (LH, HL, HH) = pywt.dwt2(Cb, wavelet=mother_wavelet)
#     Cb = LL
#
#     plt.suptitle("Poziomy DWT - operacje na składowej Cb")
#
#     # 4 obrazy na poziom
#     plt.subplot(dwt_level, 4, plot_idx); plt.imshow(LL, cmap='gray')
#     plt.title(f"Poziom {current_level}: LL")
#     plt.colorbar()
#     plot_idx += 1
#     plt.subplot(dwt_level, 4, plot_idx); plt.imshow(np.abs(LH), cmap='gray')
#     plt.title("LH")
#     plt.colorbar()
#     plot_idx += 1
#     plt.subplot(dwt_level, 4, plot_idx); plt.imshow(np.abs(HL), cmap='gray')
#     plt.title("HL")
#     plt.colorbar()
#     plot_idx += 1
#     plt.subplot(dwt_level, 4, plot_idx); plt.imshow(np.abs(HH), cmap='gray')
#     plt.title("HH")
#     plt.colorbar()
#     plot_idx += 1
#
# plt.tight_layout()
# plt.show()

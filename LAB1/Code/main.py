import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft

def inverse_fft(magnitude_log, complex_moduo_1):
    img_fft = complex_moduo_1 * np.exp(magnitude_log)
    img_filtered = np.abs(np.fft.ifft2(img_fft))

    return img_filtered

if __name__ == '__main__':

    # UCITAVANJE ULAZNE SLIKE
    img = cv2.imread("./slika_0.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # PRIKAZ ULAZNE SLIKE
    plt.imshow(img, cmap='gray')
    plt.show()

    img_fft = fft(img)
    img_fft_mag = np.abs(
        img_fft)

    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)

    # MAGNITUDA PRE UKLANJANJA SUMA
    plt.imshow(img_fft_log)
    plt.show()
    img_fft_log_scaled = ((img_fft_log - np.min(img_fft_log)) / (np.max(img_fft_log) - np.min(img_fft_log) + 1e-6) * 255).astype(np.uint8)

    cv2.imwrite('fft-before.png', img_fft_log_scaled)


    img_fft_log[156, 236] = 0
    img_fft_log[356, 276] = 0
    img_fft_log[156, 276] = 0
    img_fft_log[356, 236] = 0


    # MAGNITUDA NAKON UKLANJANJA SUMA
    plt.imshow(img_fft_log)
    plt.show()
    img_fft_log_scaled = (
                (img_fft_log - np.min(img_fft_log)) / (np.max(img_fft_log) - np.min(img_fft_log) + 1e-6) * 255).astype(
        np.uint8)

    cv2.imwrite('fft-after.png', img_fft_log_scaled)

    # PRIKAZ ULAZNE SLIKE BEZ SUMA
    img_filtered = inverse_fft(img_fft_log, img_mag_1)
    plt.imshow(img_filtered, cmap='gray')
    plt.show()
    cv2.imwrite('slika_0_final.png', img_filtered)

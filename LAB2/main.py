import numpy as np
import cv2
import matplotlib.pyplot as plt


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


if __name__ == '__main__':
    img = cv2.imread("./coins.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    plt.imshow(img_gray)
    plt.show()

    _, mask_circles = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV)

    plt.imshow(mask_circles)
    plt.show()

    mask_filtered = cv2.morphologyEx(mask_circles, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

    plt.imshow(mask_filtered)
    plt.show()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturated = hsv_img[:,:,1]
    _, marker = cv2.threshold(saturated, 80, 255, cv2.THRESH_BINARY)
    marker_filtered = cv2.morphologyEx(marker, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    plt.imshow(marker_filtered)
    plt.show()

    final_reconstructed_image = morphological_reconstruction(marker_filtered, mask_filtered)
    final_closed = cv2.morphologyEx(final_reconstructed_image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    final = cv2.morphologyEx(final_closed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    final_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    plt.imshow(final)
    plt.show()

    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    coin = cv2.bitwise_and(final_rgb, img)

    # bgr_coin = cv2.cvtColor(coin, cv2.COLOR_GRAY2BGR)
    # rgb_coin = cv2.cvtColor(bgr_coin, cv2.COLOR_BGR2RGB)
    # coin_final = cv2.bitwise_and(rgb_coin,img)

    plt.figure(0)
    plt.subplot(221)
    plt.imshow(img)

    plt.subplot(222)
    plt.imshow(coin)
    plt.show()


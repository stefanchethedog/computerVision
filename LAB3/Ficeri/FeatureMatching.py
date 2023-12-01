import cv2
import numpy as np
import cv2 as cv
import cv2 as cv
import numpy as np

img1 = cv.imread("1.JPG")
img2 = cv.imread("2.JPG")
img3 = cv.imread("3.JPG")


def trim(frame):
    """
    Brise vrste i kolone koje su skroz crne (ciji su pixeli skroz crni)
    :param frame: konacna slika
    :return:
    """
    # da li je gore ceo red crn, ako jeste onda je sum = 0
    if not np.sum(frame[0]):
        return trim(frame[1:])
    # donja vrsta skroz crna
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    # leva strana
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    # desna strana
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def stichTwoImages(im1, im2):
    detector = cv.SIFT_create()
    keyPoints1, descriptor1 = detector.detectAndCompute(im1, None)
    keyPoints2, descriptor2 = detector.detectAndCompute(im2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptor1, descriptor2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keyPoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keyPoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    else:
        print(f"Not enought matches are found - {(len(good) / MIN_MATCH_COUNT)}")
        exit(1)
    dst = cv.warpPerspective(im2, M, (im1.shape[1] + im2.shape[1], im2.shape[0] + 100))
    """
        dodajem 100 na height zato sto kad se slika transformise postaje veca i gube se pixeli
        na kraju trimujem crne redove sa svih strana da bi dobio sliku kakva treba da bude
    """
    dst[0:im1.shape[0], 0:im1.shape[1]] = im1
    return dst


desna = stichTwoImages(img2, img3)
konacna = stichTwoImages(img1, desna)
output = trim(konacna)

# cuvanje slika
cv.imshow("output", output)
cv.imwrite("output.jpg", output)

cv.waitKey(0)
cv.destroyAllWindows()


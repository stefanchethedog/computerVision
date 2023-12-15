import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt


def pyramid(image, scale=1.5, minSize=(30, 30)):
    yield image, image.shape[0], image.shape[1]
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image, image.shape[0], image.shape[1]


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


img = cv2.imread("input_img.png")
img = img[215:939, 28:1471]
(winW, winH) = (180, 180)

rows = open("./synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

for layer, layer_x, layer_y in pyramid(img, scale=2):
    for (x, y, window) in sliding_window(layer, stepSize=180, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        blob = cv2.dnn.blobFromImage(window, 1, (224, 224), (100, 130, 123))

        net = cv2.dnn.readNetFromCaffe("./bvlc_googlenet.prototxt", "./bvlc_googlenet.caffemodel")
        net.setInput(blob)
        preds = net.forward()
        idxs = np.argsort(preds[0])[::-1][:5]
        for (i, idx) in enumerate(idxs):
            if preds[0][idx] > 0.5:
                if i == 0:
                    skaliranje = int(img.shape[0] / layer_x)
                    if "cat" in classes[idx]:
                        text = "CAT"
                        boja = (0, 0, 255)
                    elif "dog" in classes[idx] or classes[idx] == "toy poodle":
                        text = "DOG"
                        boja = (0, 255, 255)
                    else:
                        continue
                    cv2.putText(img, text, (x * skaliranje + 5, y * skaliranje + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, boja, 2)
                    cv2.rectangle(img, (x * skaliranje, y * skaliranje), (x * skaliranje + winW * skaliranje - 5, y * skaliranje + winH *
                                                                          skaliranje - 5), boja, 2)
            # time.sleep(0.1)
        cv2.waitKey(1)

cv2.imshow("output.png", img)
cv2.imwrite("output.png",img)
cv2.waitKey(0)
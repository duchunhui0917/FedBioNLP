import matplotlib.pyplot as plt
import numpy as np
import cv2


def bbox_to_rect(bbox, color):
    return plt.Rectangle(xy=(bbox[0], bbox[1]),
                         width=bbox[2] - bbox[0],
                         height=bbox[3] - bbox[1],
                         fill=False,
                         edgecolor=color,
                         linewidth=2)


def numpy_to_img(x, path):
    x = 1.0 / (1 + np.exp(-1 * x))
    x = np.round(x * 255)
    cv2.imwrite(path, x)
    img = cv2.imread(path)
    cv2.imshow('image', img)

import numpy as np
from skimage import io
from skimage.transform import resize


def imread(path):
    return io.imread(path)/255.0


# got from https://github.com/machrisaa
def imread2vgg(path, targetH=224):
    img = imread(path)

    W, H, _ = img.shape
    S = min(W, H)

    W, H = int((W-S)/2), int((H-S)/2)
    return resize(img[W:W+S, H:H+S, :], (targetH, targetH))


def imbatch1(img):
    return np.expand_dims(img, axis=0)
import cv2
import numpy as np
from skimage.feature import hog

def hog_features(images):
    print("Start calculating HOG features")
    hog_features = []
    for i in range(len(images)):
        img = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        features, _ = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm="L2-Hys", visualize=True)

        hog_features.append(features)
    print("Finish calculating HOG features")
    return np.array(hog_features)

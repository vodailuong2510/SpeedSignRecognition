import cv2
import numpy as np
from sklearn.svm import SVC

def detect_sign(images):
    hsv_images = cv2.cvtColor(images, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_images, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_images, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def SVC_training(images, labels, kernel:str):
    print("Start training")
    clf = SVC(kernel=kernel)
    clf.fit(images, labels)
    print("Finish training")
import cv2
from skimage.feature import hog
from .models import detect_sign

def hog_features(image):
    features, _ = hog(image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
    return features

def roi_features(images):
    roi_images = []

    for image in images:
        contours, _ = detect_sign(image)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = image[y:y+h, x:x+w]
            try:
                features = hog_features(roi)
                roi_images.append(features)
            except:
                continue
        
    return images
import cv2
from pathlib import Path
import numpy as np
import re
import albumentations as A
from imblearn.over_sampling import RandomOverSampler

def read_data(path : str):
    print("Start reading data")

    img_path = Path(path + "/images")
    img_files = sorted(img_path.iterdir(), key=lambda x: int(re.search(r'\d+', x.name).group()))

    label_path = Path(path +"/labels.txt")
    with open(label_path, 'r') as file:
        labels = [int(line.strip()) for line in file]

    images = []

    for img_path in img_files:
        if img_path.suffix in ['.jpg', '.png', '.jpeg']:
            img = cv2.imread(str(img_path))

            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

    print("Read data successfully")
    return images, labels

def preprocessing_augment(images, labels, resize, augment = False, weight = 5):
    print("Start preprocessing")
    if augment:
        augmentation = A.Compose([
            A.Rotate(limit=15, p=0.5),  
            A.GaussianBlur(blur_limit=(3, 7), p=0.5), 
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5), 
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, p=0.5),
        ])

        augmented_images = []
        augmented_labels = []

        for img, label in zip(images, labels):
            img = img.astype(np.uint8) 
            for _ in range(weight): 
                aug_img = augmentation(image=img)["image"]
                augmented_images.append(aug_img)
                augmented_labels.append(label)

            augmented_images.append(img)
            augmented_labels.append(label)

        augmented_images = [cv2.resize(img, resize) for img in augmented_images]
        augmented_images = [cv2.GaussianBlur(img, (5, 5), 0) for img in augmented_images]
        augmented_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in augmented_images]
        augmented_images = [cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) for img in augmented_images]

        print("Finish preprocessing")
        return np.array(augmented_images), np.array(augmented_labels)

    else:
        images = [cv2.resize(img, resize) for img in images]
        images = [cv2.GaussianBlur(img, (5, 5), 0) for img in images]
        images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images]
        images = [cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) for img in images]

        print("Finish preprocessing")
        return np.array(images), np.array(labels)
    
def over_sampling(images, labels, resize):
    images = [cv2.resize(img, resize) for img in images]
    images = np.array(images)

    images_flat = images.reshape(len(images), -1)

    ros = RandomOverSampler(random_state=22520834)
    
    images_resampled, labels_resampled = ros.fit_resample(images_flat, labels)

    images_resampled = images_resampled.reshape(-1, resize[0], resize[1], 3)

    return np.array(images_resampled), np.array(labels_resampled)
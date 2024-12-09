import cv2
from pathlib import Path
import numpy as np
import re

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

def preprocessing(images, labels, resize):
    for i in range(len(images)):
        images[i] = cv2.resize(images[i], resize)
        images[i] = cv2.GaussianBlur(images[i], (5, 5), 0)
        images[i] = cv2.normalize(images[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    images = np.array(images) #/ 255.0
    labels = np.array(labels)

    return images, labels
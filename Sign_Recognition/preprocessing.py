import wget
import zipfile
import cv2
from pathlib import Path

def download(link:str):
    wget.download(link)

    print("\nDownloaded data successfully")

def unzip(zip_path:str, extract_path:str):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print("Extracted files successfully")

def read_data(path : str, resize):
    data_path = Path(path)
    data_folders= data_path.iterdir()

    images = []
    labels = []

    for i, folder in enumerate(data_folders):
        label = i
        for img_path in folder.iterdir():
            if img_path.suffix in ['.jpg', '.png']:
                img = cv2.imread(str(img_path))

                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, resize)
                    images.append(img)
                    labels.append(label)

    return images, labels

def preprocessing_images(images):
    images = images / 255.0
    return images

def augment(images):
    pass
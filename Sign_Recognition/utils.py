import wget
import zipfile
import random
import matplotlib.pyplot as plt
random.seed(22520834)

def plot_images(images, labels, class_names, title="", num_images=5):
    random_indices = random.sample(range(len(images)), num_images)

    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(random_indices):
        plt.subplot(num_images//5 + 1, 5, i+1)
        plt.imshow(images[idx], cmap = "gray")
        plt.title(class_names[labels[idx]])
        plt.axis("off")
    
    plt.suptitle(title)
    plt.show()

def download(link:str):
    print("Start downloading data")

    wget.download(link)

    print("\nDownloaded data successfully")

def unzip(zip_path:str, extract_path:str):
    print("Start extracting files")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print("Extracted files successfully")
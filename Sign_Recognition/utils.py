import pandas as pd
from pathlib import Path
import random
import matplotlib.pyplot as plt

def plot_images(images, labels, class_names, title="", num_images=5):
    random_indices = random.sample(range(len(images)), num_images)

    plt.figure(figsize=(15, 15))
    for i, idx in enumerate(random_indices):
        plt.subplot(num_images//5 + 1, 5, i+1)
        plt.imshow(images[idx])
        plt.title(class_names[labels[idx]])
        plt.axis("off")
    
    plt.suptitle(title)
    plt.show()
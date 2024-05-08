import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def load_images(source_path: str, count: int):
    images = []
    for i in range(count):
        image_file = os.listdir(source_path)[i]
        image_file_path = os.path.join(source_path, image_file)

        with Image.open(image_file_path) as img:
            images.append(img.copy())

    return images


def plot_images(images: list, title: str):
    plt.figure()
    plt.suptitle(title, fontsize=14)
    for i, img in enumerate(images[:4]):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.title(f"Image {i+1}", fontsize=12)
        plt.axis("off")
    plt.show()


def image_stats(images):
    image_sizes = [img.size for img in images]
    color_channels = [img.mode for img in images]

    # Analyze image sizes
    unique_sizes = set(image_sizes)
    if len(unique_sizes) > 1:
        print("Varying image sizes:", unique_sizes)
    else:
        print("Consistent image size:", unique_sizes.pop())

    # Analyze color channels
    channel_count = Counter(color_channels)
    print("Color channel modes:", channel_count)


def pixel_intensity(images):
    # Assuming the images list contains PIL Image objects in RGB format
    pixel_values = np.concatenate(
        [np.array(img).reshape(-1, 3) for img in images if img.mode == "RGB"], axis=0
    )

    # Separate channels
    red, green, blue = pixel_values[:, 0], pixel_values[:, 1], pixel_values[:, 2]

    # Plot histograms for each color channel
    plt.figure(figsize=(10, 5))
    plt.hist(red, bins=256, color="red", alpha=0.6, label="Red Channel")
    plt.hist(green, bins=256, color="green", alpha=0.6, label="Green Channel")
    plt.hist(blue, bins=256, color="blue", alpha=0.6, label="Blue Channel")
    plt.legend()
    plt.title("Pixel Intensity Distribution Across Color Channels")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

import os
import json
import numpy as np
from PIL import Image
from numpy import expand_dims
from collections import Counter
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


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


def pixel_intensity(images, title: str):
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
    plt.title(f"Pixel Intensity Distribution Across Color Channels - {title}")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()


def build_sequential_model(
    layers: list, optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
):
    model = Sequential(layers)

    model.compile(optimizer, loss, metrics)

    return model


def load_augmented_data(
    base_path: str,
    train_data_config={},
    validation_data_config={},
    batch_size=10,
    target_size=(32, 32),
):

    updated_train_data_config = {
        "rescale": 1.0 / 255,
        "rotation_range": 40,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "fill_mode": "nearest",
        **train_data_config,
    }

    train_data_gen = ImageDataGenerator(**updated_train_data_config)
    train_generator = train_data_gen.flow_from_directory(
        f"{base_path}/train/",
        batch_size=batch_size,
        target_size=target_size,
    )

    updated_validation_data_config = {
        "rescale": 1.0 / 255,
        **validation_data_config,
    }

    validation_data_gen = ImageDataGenerator(**updated_validation_data_config)
    validation_generator = validation_data_gen.flow_from_directory(
        f"{base_path}/test",
        batch_size=batch_size,
        target_size=target_size,
    )

    return train_generator, validation_generator


def save_model_history(model, history, model_name):
    model.save(f"./models/{model_name}/model")

    with open(f"./models/{model_name}/history.json", "w") as f:
        json.dump(history.history, f)


def load_model_history(model_name):
    model = None
    history = None
    model_history_loaded = True

    try:
        model = load_model(f"./models/{model_name}/model")
        with open(f"./models/{model_name}/history.json", "r") as f:
            history = json.load(f)
    except:
        model_history_loaded = False
        print("Model/history not found.")

    return model_history_loaded, model, history


def save_plot(dir_name, filename):
    directory = f"plots/{dir_name}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    filepath = os.path.join(directory, filename)
    plt.savefig(filepath)


def plot_performance_curves(model_history, model_name: str):
    model_title = model_name.replace("_", " ").title()

    # Loss Curves
    plt.figure(figsize=(8, 6))
    plt.plot(model_history["loss"])
    plt.plot(model_history["val_loss"], ls="--")
    plt.legend(["Training Loss", "Testing Loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve | {model_title}")
    save_plot(model_name, "loss_curve.png")

    # Accuracy Curves
    plt.figure(figsize=(8, 6))
    plt.plot(model_history["accuracy"])
    plt.plot(model_history["val_accuracy"], ls="--")
    plt.legend(["Training Accuracy", "Testing Accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    save_plot(model_name, "accuracy_curve.png")


def plot_augmented_images(img_path):
    # load the image
    img = load_img(img_path)
    # Convert to numpy array
    data = img_to_array(img)
    # Expand dimension to one sample
    samples = expand_dims(data, 0)
    # Create image data augmentation generator
    data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    # Prepare iterator
    it = data_gen.flow(samples, batch_size=1)
    plt.figure(figsize=(12, 10))
    # Generate samples and plot
    for i in range(9):
        # define subplot
        plt.subplot(330 + 1 + i)
        # Generate batch of images
        batch = it.next()
        # Convert to unsigned integers for viewing
        image = batch[0].astype("uint8")
        # Plot raw pixel data
        plt.imshow(image)
    # Show the figure
    plt.show()

import os
import json
import numpy as np
from PIL import Image
from numpy import expand_dims
from collections import Counter
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.metrics import Precision, Recall
from keras.utils import image_dataset_from_directory
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)
import seaborn as sns
from keras.utils import set_random_seed

# What's the meaning of life, the universe and everything!?
set_random_seed(42)


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
    layers: list,
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", Precision(), Recall()],
    no_compile=False,
):
    model = Sequential(layers)

    if not no_compile:
        model.compile(optimizer, loss, metrics)

    return model


def load_data(
    base_path: str,
    augmented=True,
    train_data_config=None,
    validation_data_config=None,
    batch_size=10,
    target_size=(32, 32),
    class_mode="binary",
):
    if augmented:
        updated_train_data_config = {
            "rescale": 1.0 / 255,
            "rotation_range": 40,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "horizontal_flip": True,
            "fill_mode": "nearest",
        }

        if train_data_config is not None:
            updated_train_data_config = train_data_config

        train_data_gen = ImageDataGenerator(**updated_train_data_config)
        train_generator = train_data_gen.flow_from_directory(
            f"{base_path}/train/",
            batch_size=batch_size,
            target_size=target_size,
            class_mode=class_mode,
        )

        updated_validation_data_config = {
            "rescale": 1.0 / 255,
        }

        if validation_data_config is not None:
            updated_train_data_config = validation_data_config

        validation_data_gen = ImageDataGenerator(**updated_validation_data_config)
        validation_generator = validation_data_gen.flow_from_directory(
            f"{base_path}/test/",
            batch_size=batch_size,
            target_size=target_size,
            class_mode=class_mode,
            shuffle=False,
        )

        return train_generator, validation_generator
    else:
        train_ds = image_dataset_from_directory(
            f"{base_path}/train/",
            image_size=(target_size[0], target_size[1]),
            batch_size=batch_size,
        )

        validation_ds = image_dataset_from_directory(
            f"{base_path}/test/",
            image_size=(target_size[0], target_size[1]),
            batch_size=batch_size,
        )

        return train_ds, validation_ds


def save_model_history(model, history, model_name):
    model.save(f"./models/{model_name}/model")

    with open(f"./models/{model_name}/history.json", "w") as f:
        json.dump(history, f)


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
    plt.savefig(
        filepath,
        bbox_inches="tight",
    )


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
    plt.title(f"Accuracy Curve | {model_title}")
    save_plot(model_name, "accuracy_curve.png")

    # Precision Curves
    if "precision" in model_history:
        plt.figure(figsize=(8, 6))
        plt.plot(model_history["precision"])
        plt.plot(model_history["val_precision"], ls="--")
        plt.legend(["Training Precision", "Testing Precision"])
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.title(f"Precision Curve | {model_title}")
        save_plot(model_name, "precision_curve.png")

    # Recall Curves
    if "recall" in model_history:
        plt.figure(figsize=(8, 6))
        plt.plot(model_history["recall"])
        plt.plot(model_history["val_recall"], ls="--")
        plt.legend(["Training Recall", "Testing Recall"])
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.title(f"Recall Curve | {model_title}")
        save_plot(model_name, "recall_curve.png")


def plot_augmented_image(img):
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
        plt.axis("off")

    # Show the figure
    plt.show()


def plot_generator_images(generator, count):
    # Retrieve a batch of images
    images, _ = next(generator)

    num_images = min(count, len(images))

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def load_test_data(
    base_path,
    augmented=True,
    data_config=None,
    batch_size=10,
    target_size=(32, 32),
    class_mode="binary",
):
    if augmented:
        updated_data_config = {
            "rescale": 1.0 / 255,
        }

        if data_config is not None:
            updated_data_config = data_config

        data_gen = ImageDataGenerator(**updated_data_config)

        generator = data_gen.flow_from_directory(
            base_path,
            batch_size=batch_size,
            target_size=target_size,
            class_mode=class_mode,
            shuffle=False,
        )

        return generator
    else:
        ds = image_dataset_from_directory(
            base_path,
            image_size=(target_size[0], target_size[1]),
            batch_size=batch_size,
        )
        return ds


def plot_test_metrics(classifier, generator, model_name, test_dataset_name):
    model_title = model_name.replace("_", " ").title()
    filename_suffix = test_dataset_name.replace(" ", "_").lower()

    predictions = classifier.predict(generator)

    y_pred = (predictions > 0.5).astype("int32").flatten()
    y_true = generator.classes

    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred)}")
    print(f"Classification Report: \n {classification_report(y_true, y_pred)}")

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve | {model_title} | {test_dataset_name}")
    save_plot(
        model_name,
        f"roc_curve_{filename_suffix}.png",
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=generator.class_indices,
        yticklabels=generator.class_indices,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix | {model_title} | {test_dataset_name}")
    save_plot(
        model_name,
        f"confusion_matrix_{filename_suffix}.png",
    )

    return roc_auc_score(y_true, y_pred), classification_report(
        y_true, y_pred, output_dict=True
    )


def save_test_metrics(
    accuracy,
    loss,
    precision,
    recall,
    roc_auc_score,
    classification_report,
    test_dataset_name,
    model_name,
):
    filename_suffix = test_dataset_name.replace(" ", "_").lower()

    metrics = {
        "test_dataset_name": test_dataset_name,
        "accuracy": accuracy,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "classification_report": classification_report,
        "roc_auc": roc_auc_score,
    }

    with open(
        f"models/{model_name}/testing_metrics_{filename_suffix}.json", "w"
    ) as json_file:
        json.dump(metrics, json_file, indent=4)

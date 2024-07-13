# Work Log

This log is to record the work done on the weekly basis.

## Week 1

-   Identified problem statement
-   Created project outline
-   Created outline presentation
-   Literature review

## Week 2

-   Working on Anaconda environment setup
-   Created environment `tf-gpu` to run tensorflow on GPU
-   Collected datasets
    -   [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images?resource=download)
    -   [AI Art](https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset)
    -   [Fake cats and dogs](https://www.kaggle.com/datasets/mattop/ai-cat-and-dog-images-dalle-mini/code)
-   Organized datasets

## Week 3

-   Created basic CNN model to analyze preliminary performance and training time across various platforms
-   Explored multiple ways to efficiently upload large dataset to Google Drive to work with Colab
    -   Written multiple scripts to upload data
    -   Uploaded CIFAKE data to Drive
    -   Conclusion
        -   Time consuming to upload data to drive
        -   Also Colab is not very efficient
-   Explored cloud solution for faster computes
    -   Training time for one epoch
        -   Kaggle (2 GPUs) - 15mins
        -   Google Colab (1 GPU) - 4+hrs (Not able to consume GPU)
        -   Local (1 GPU) - 9mins

## Week 4

-   Created utils file for utility functions
    -   `load_images` - function to load images
    -   `plot_images` - function to plot 4 images
    -   `images_stats` - function to log the image statistics
    -   `pixel_intensity` - function to plot pixel intensity of the images
-   Exploratory data analysis on CIFAKE dataset

## Week 5

-   Exploratory data analysis on AI art dataset
-   Merged CIFAKE and AI Art dataset for testing purpose
-   Created basic model to explore preliminary performance for AI art dataset
-   Exploring how to handle varying size images while training models
-   Exploring various approaches for dimensionality reduction in images

## Week 6

-   Continuation of exploratory data analysis on AI art dataset
    -   Analysis on efficient images resizing
    -   Created utils for reusability
-   Implemented model and history saving
    -   Significantly help to avoid retraining the models
-   Created reusable model training template and utility functions
    -   Reduce code for future models
-   Reused template for AI art basic model
-   Reused template for CIFAKE basic model
-   Created model performance log

## Week 7

-   Mid term report and presentation preparation

## Week 8

-   Identified the models that need to be developed
-   Created basic template for model training

## Week 9 - 10

-   Started model training
-   Trained around 30 models with various architectures

## Week 11 - 12

-   Hyperparameter tuning of models
-   Creating model comparison template

## Week 13

-   Model comparison
-   Creating model testing template

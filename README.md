# Dental Image Classification with CNN (PyTorch)

This project applies a Convolutional Neural Network (CNN) to classify dental images into categories based on visual features. The model is built from scratch using PyTorch and trained on a custom dataset of dental images. The entire process, from data preprocessing to model evaluation, is done in a single notebook on Kaggle using a Tesla P100 GPU.

## Project Overview

* Preprocesses dental images (resizing, normalization, and augmentation)
* Visualizes class distribution across training, validation, and test sets
* Defines and trains a CNN model without pretrained weights
* Evaluates model performance on a held-out test set
* Experiments with deeper CNN architectures and optimizer improvements

## Dataset

The dataset consists of images organized in a folder structure compatible with `torchvision.datasets.ImageFolder`:

```
Teeth_Dataset/
├── Training/
├── Validation/
└── Testing/
```

Each subfolder contains images sorted into class directories. During preprocessing, unnecessary folders such as "out", "output", or "outputs" were excluded.

## Model Architecture

The best-performing model is a medium-depth CNN consisting of:

* 3 convolutional blocks with increasing filters (32 → 64 → 128)
* Batch normalization and dropout for regularization
* Global average pooling followed by a fully connected classifier

The model is trained using the AdamW optimizer and cross-entropy loss.

## Requirements

This project runs in a Kaggle environment and uses:

* Python 3.x
* PyTorch
* Torchvision
* Matplotlib
* Seaborn
* Pandas
* NumPy

No additional installations are needed when run on Kaggle.

## How to Use

1. Upload your dataset to Kaggle as a private dataset
2. Load the dataset into a notebook using Kaggle's `/kaggle/input/...` path
3. Run the notebook to:

   * Preprocess and visualize the data
   * Define and train the CNN model
   * Evaluate test accuracy

The model can be trained in under 5 minutes on a Kaggle GPU (P100) for 20 epochs.

## Results

The baseline model achieved low accuracy due to its shallow structure. After upgrading to a deeper CNN with better regularization and using the AdamW optimizer, validation and test accuracy improved significantly. Final performance may vary depending on dataset quality and class balance.


## License

This project is for educational purposes. Feel free to fork or modify it for personal use.

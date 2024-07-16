# Image Processing and GAN Training Pipeline
This repository contains a complete image processing and GAN (Generative Adversarial Network) training pipeline. The code handles various stages of image preprocessing, data augmentation, model definition, and training.

## Pipeline Overview
### Loading and Extracting Zip File
Extracts images from a zip file into a specified directory.

### Loading Images
Loads images from the extracted directory and converts them to RGB format.

### CLAHE Histogram Equalization
Enhances image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

### Resizing and Normalizing Images
Resizes images to 150x212 pixels and normalizes pixel values to the range [-1, 1].

### Data Augmentation
Applies random transformations like rotation, shifting, shearing, and flipping to augment the dataset.

### GAN Model Definition
Defines the generator and discriminator models using TensorFlow and Keras.

### Training the GAN
Trains the GAN with the processed and augmented images, saving checkpoint models and generated images at intervals.

## Installation
Clone the repository and install the required libraries:
git clone https://github.com/Anshuman-Samantaray/Generative-Adversarial-Network.git
cd Generative-Adversarial-Network
pip install -r requirements.txt

# Image Processing and GAN Training Pipeline

This repository contains a complete image processing and GAN (Generative Adversarial Network) training pipeline. The code handles various stages of image preprocessing, data augmentation, model definition, and training, with additional features to improve training stability and evaluation.

## Pipeline Overview

### Loading and Extracting Zip File
Extracts images from a zip file into a specified directory.

### Loading Images
Loads images from the extracted directory and converts them to RGB format.

### CLAHE Histogram Equalization
Enhances image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

### Resizing and Normalizing Images
Resizes images to 152x212 pixels and normalizes pixel values to the range [-1, 1].

### Data Augmentation
Applies random transformations like rotation, shifting, shearing, and flipping to augment the dataset.

### GAN Model Definition
Defines the generator and discriminator models using TensorFlow and Keras.

### Training the GAN
Trains the GAN with the processed and augmented images. The training process includes:
- **Checkpointing:** Saving model checkpoints and generated images at intervals.
- **Inception Score Calculation:** Evaluating the quality of generated images and monitoring the Inception Score at regular intervals.
- **Learning Rate Adjustment:** Adjusting the learning rate if the Inception Score does not improve to tackle potential false loop issues. 

### Updated Features in `submission.ipynb`

This repository includes three `.ipynb` files:

1. **`submission.ipynb`:** This file contains the latest updates to the image processing and GAN training pipeline. It includes the following new features:
   - **Inception Score Calculation:** Added a method to evaluate the quality of generated images using the Inception Score metric.
   - **Training Loop with Quality Check:** Updated the training loop to check the Inception Score at regular intervals (`inception_check_freq` epochs) and reduce the learning rate if the score does not improve for a specified number of epochs (`patience` epochs).
   - **Learning Rate Adjustment:** Incorporated a mechanism to adjust the learning rate based on the Inception Score to help the model escape potential false looping situations.
   - **Enhanced Error Handling:** Improved robustness with additional error handling for file operations and image processing.

2. **`code.ipynb`:** Contains the GAN pipeline without the advanced features introduced in the latest version.

3. **`code_with_prints.ipynb`:** Contains the code of final_code.ipynb but has the facility to show original, resized and equalized images.

## How to Use

1. **Prepare the Data:**
   - Place the dataset zip file in the specified directory and update the `zip_path` and `extract_path` variables in the code.

2. **Run the Notebook:**
   - Execute the `submission.ipynb` file in a Jupyter environment to perform the entire pipeline, including image preprocessing, data augmentation, GAN training, and evaluation.

3. **Monitor Training:**
   - The training process will display the Inception Score and save generated images and model checkpoints periodically.

4. **Adjust Hyperparameters:**
   - You can modify hyperparameters such as `EPOCHS`, `BATCH_SIZE`, and `inception_check_freq` in the `submission.ipynb` file to fit your specific needs.

## Example Code Snippet

```python
# Define GAN models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(initial_learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(initial_learning_rate)

# Training the model
train(train_dataset, EPOCHS, inception_check_freq=10, patience=20)


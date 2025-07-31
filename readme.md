End-to-End Fashion-MNIST Analysis
Project Overview

This repository documents a comprehensive exploration of the Fashion-MNIST dataset, demonstrating a full machine learning workflow from basic classification to advanced deep learning techniques. The project is divided into four main parts, each contained within its own Jupyter Notebook.
1. Scikit-Learn Classification (Baseline Models)

Notebook: notebooks/1_Scikit-Learn_Classification.ipynb

This notebook establishes a baseline performance using traditional machine learning models from Scikit-learn.

    Models Trained: Logistic Regression, Random Forest Classifier.

    Techniques: Data scaling (StandardScaler), cross-validation, and dimensionality reduction with PCA.

    Key Finding: The Random Forest model was selected as the best performer, providing a solid baseline before moving to deep learning.

2. Convolutional Neural Network (CNN)

Notebook: notebooks/2_CNN_Classification.ipynb

This notebook implements a Convolutional Neural Network (CNN), the state-of-the-art approach for image classification tasks.

    Architecture: A deep CNN with multiple Conv2D and MaxPooling2D layers, followed by Dense layers for classification.

    Techniques: TensorFlow/Keras, data standardization, and dropout for regularization.

    Result: Achieved a test accuracy of ~88.5%, demonstrating the power of CNNs in recognizing spatial patterns in images.

3. Dense Neural Network (DNN)

Notebook: notebooks/3_DNN_Classification.ipynb

This notebook builds a fundamental deep learning model, a Dense Neural Network (DNN), to compare its performance against the specialized CNN.

    Architecture: A sequential model with a Flatten layer and multiple Dense (fully-connected) layers.

    Techniques: TensorFlow/Keras, data scaling (0-1 normalization), and validation set for monitoring training.

    Result: Achieved a test accuracy of ~88.0%. While very strong, this result highlights the advantage of using CNNs for image data.

4. Unsupervised Learning with an Autoencoder (AE)

Notebook: notebooks/4_Autoencoder.ipynb

This notebook explores unsupervised learning by building a stacked autoencoder to compress and reconstruct the fashion images.

    Architecture: An encoder-decoder structure built with Dense layers. The encoder compresses each 784-pixel image into a 30-dimensional vector.

    Techniques: Unsupervised training, feature learning, and t-SNE for visualizing the learned feature space.

    Result: The model successfully reconstructed images with a high pixel accuracy (~93.5%) and the t-SNE plot showed that the autoencoder learned to group similar fashion items together in the compressed space.

How to Run

    Clone this repository.

    Install the required libraries: pip install -r requirements.txt

    Navigate to the notebooks/ directory and open any of the Jupyter Notebooks to explore that part of the project.

Acknowledgements

This project uses the Fashion-MNIST dataset provided by Zalando Research.
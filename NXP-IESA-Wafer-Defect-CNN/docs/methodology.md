# Methodology

## 1. Data Acquisition and Augmentation
Given the scarcity of public semiconductor defect datasets, we employed a physics-informed synthetic data generation strategy.
- **Base Texture**: Simulated silicon wafer grain using Perlin noise and Gaussian filtering.
- **Defect Modeling**:
    - **Geometric Shapes**: Lines, circles, and polygons were drawn to mimic circuit patterns.
    - **Physics Simulation**: Added Poisson noise to simulate electron shot noise common in SEM imaging.
    - **Blur**: Applied a Point Spread Function (PSF) approximation to simulate electron beam interaction volume.
- **Augmentation**: The training pipeline uses `tf.keras.utils.image_dataset_from_directory` which handles batching and resizing. Random transformations (flip, rotation) can be added as layers within the model or preprocessing steps.

## 2. Preprocessing
- **Grayscale Conversion**: SEM images are treated as single-channel intensity maps.
- **Resizing**: All images are resized to 256x256 pixels.
- **Rescaling**: Pixel values are rescaled from [0, 255] to [0.0, 1.0] using a `Rescaling` layer.

## 3. Training Strategy
- **Framework**: TensorFlow / Keras.
- **Optimizer**: Adam (Adaptive Moment Estimation).
- **Loss Function**: Sparse Categorical Crossentropy (suitable for integer-labeled multi-class classification).
- **Regularization**: Dropout layers (p=0.5) to prevent overfitting.
- **Validation**: 30% of the dataset is held out for validation during training.
- **Metrics**: Accuracy, Mean Absolute Error (MAE), Mean Squared Error (MSE).

## 4. Evaluation Metrics
We evaluate the model using:
- **Classification Report**: Precision, Recall, F1-Score for each class.
- **Confusion Matrix**: Visual heatmap of true vs. predicted classes.
- **Inference Confidence**: Probability scores for predictions.

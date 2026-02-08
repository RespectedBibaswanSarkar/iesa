import tensorflow as tf
from tensorflow.keras import layers, models

def create_wafer_defect_model(input_shape=(256, 256, 1), num_classes=6):
    """
    Creates a detailed CNN model for Wafer Defect Classification.
    Based on the architecture provided by the user.
    """
    model = models.Sequential([
        # Input and Rescaling
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),

        # Block 1
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Block 2
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Block 3
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),

        # Dense Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    model = create_wafer_defect_model()
    model.summary()

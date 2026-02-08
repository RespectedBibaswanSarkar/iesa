import tensorflow as tf
from tensorflow.keras import layers, models

def create_wafer_defect_model(input_shape=(64, 64, 1), num_classes=6):
    """
    Creates a lightweight CNN model for Wafer Defect Classification.
    Architecture:
    - Rescaling 1./255
    - Conv2D 16
    - Conv2D 32
    - GlobalAveragePooling2D
    - Dense 64
    - Dropout 0.4
    - Dense num_classes (Softmax)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    model = create_wafer_defect_model()
    model.summary()

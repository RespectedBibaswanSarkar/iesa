import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def setup_dataset(zip_path, target_dir):
    """
    Handles dataset setup: unzip if needed, check structure.
    """
    # Check if target exists and has content
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"Directory '{target_dir}' already exists and is not empty. Using existing data.")
        return

    # Check zip
    if not os.path.exists(zip_path):
        # Fallback: Check if we can generate it? 
        # For now, just warn if neither exists.
        print(f"Warning: Zip file '{zip_path}' not found and target dir is empty.")
        print("Please ensure data is generated or the zip file is present.")
        return

    print(f"Extracting '{zip_path}' to '{target_dir}'...")
    try:
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting zip: {e}")
        # Cleanup partial extraction
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        raise

def plot_training_history(history, save_dir="results"):
    """
    Plots accuracy and loss curves from Keras history object.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
    plt.close()

def plot_confusion_matrix_heatmap(y_true, y_pred, classes, save_dir="results"):
    """
    Plots confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

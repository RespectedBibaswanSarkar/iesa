import os
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model import create_wafer_defect_model
from src.utils import setup_dataset, plot_training_history, plot_confusion_matrix_heatmap

def main(args):
    # 1. Setup Data
    setup_dataset(args.zip_path, args.data_dir)
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return

    print("Loading Dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.3,
        subset="training",
        seed=42,
        image_size=(256, 256),
        batch_size=args.batch_size,
        color_mode="grayscale"
    )

    val_test_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.3,
        subset="validation",
        seed=42,
        image_size=(256, 256),
        batch_size=args.batch_size,
        color_mode="grayscale"
    )

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # Split Val/Test
    val_batches = tf.data.experimental.cardinality(val_test_ds)
    test_ds = val_test_ds.take(val_batches // 2)
    val_ds = val_test_ds.skip(val_batches // 2)

    # 2. Build Model
    model = create_wafer_defect_model(num_classes=len(class_names))
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", "mae", "mse"]
    )
    
    model.summary()

    # 3. Train
    print(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )

    # 4. Evaluation
    print("\nEvaluating on Test Set...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # 5. Save Results
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    plot_training_history(history, args.results_dir)
    plot_confusion_matrix_heatmap(y_true, y_pred, class_names, args.results_dir)

    model_save_path = os.path.join(args.model_dir, "semiconductor_defect_cnn.h5")
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # 6. Sample Predictions CSV
    print("Generating sample predictions...")
    results = []
    # Walk through a subset or all files (limit to prevent spamming if huge)
    count = 0
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                if count > 100: break # Limit for sample
                path = os.path.join(root, file)
                img = tf.keras.utils.load_img(path, color_mode="grayscale", target_size=(256, 256))
                arr = tf.keras.utils.img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)

                pred = model.predict(arr, verbose=0)
                label = class_names[np.argmax(pred)]

                results.append({"filename": file, "predicted_class": label})
                count += 1
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.results_dir, "sample_predictions.csv"), index=False)
    print("Sample predictions saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory containing class folders")
    parser.add_argument("--zip_path", type=str, default="dataset_v2.zip", help="Path to dataset zip if dir doesn't exist")
    parser.add_argument("--model_dir", type=str, default="models/trained_model", help="Where to save .h5 model")
    parser.add_argument("--results_dir", type=str, default="results", help="Where to save plots and logs")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    main(args)

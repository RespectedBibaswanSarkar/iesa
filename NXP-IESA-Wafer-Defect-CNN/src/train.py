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
    # Use 80/20 Split as requested
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(64, 64),
        batch_size=args.batch_size,
        color_mode="grayscale"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(64, 64),
        batch_size=args.batch_size,
        color_mode="grayscale"
    )

    class_names = train_ds.class_names
    NUM_CLASSES = len(class_names)
    print(f"✅ CLASSES: {class_names}")
    print(f"✅ Train batches: {len(train_ds)}") 
    print(f"✅ Val batches: {len(val_ds)}")

    # 2. Build Model
    model = create_wafer_defect_model(input_shape=(64, 64, 1), num_classes=NUM_CLASSES)
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.summary()

    # 3. Train
    print(f"\n🚀 FIXED TRAINING... for {args.epochs} epochs")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        steps_per_epoch=len(train_ds),
        validation_steps=len(val_ds),
        verbose=1
    )

    # 4. Manual Validation Test
    print("\n🔍 MANUAL VALIDATION TEST:")
    for images, labels in val_ds.take(1):
        predictions = model.predict(images, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = labels.numpy()
        manual_acc = np.mean(pred_classes == true_classes)
        print(f"✅ MANUAL val_accuracy: {manual_acc:.4f}")
        print(f"✅ Sample preds: {pred_classes[:5]}, true: {true_classes[:5]}")

    # 5. Save Results
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    plot_training_history(history, args.results_dir)
    # Generate predictions for confusion matrix on full val set
    print("\nGenerating Confusion Matrix on Validation Set...")
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    plot_confusion_matrix_heatmap(y_true, y_pred, class_names, args.results_dir)
    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=class_names))

    model_save_path = os.path.join(args.model_dir, "semiconductor_defect_cnn.h5")
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # 6. TFLite Conversion
    print("\n🔧 TFLite Conversion...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(args.model_dir, "defect_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"\n🎉 YOUR MODEL:")
    print(f"✅ Size: {size_kb:.1f} KB (PERFECT!)")
    print(f"✅ TFLite saved to: {tflite_path}")
    print("✅ HACKATHON READY!")

    # 7. Sample Predictions CSV (Optional, keeping from previous version but simplified)
    print("Generating sample predictions...")
    results = []
    count = 0
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                if count > 20: break 
                path = os.path.join(root, file)
                img = tf.keras.utils.load_img(path, color_mode="grayscale", target_size=(64, 64))
                arr = tf.keras.utils.img_to_array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)

                pred = model.predict(arr, verbose=0)
                label = class_names[np.argmax(pred)]

                results.append({"filename": file, "predicted_class": label})
                count += 1
        if count > 20: break
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.results_dir, "sample_predictions.csv"), index=False)
    print("Sample predictions saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset", help="Directory containing class folders")
    parser.add_argument("--zip_path", type=str, default="dataset_v2.zip", help="Path to dataset zip if dir doesn't exist")
    parser.add_argument("--model_dir", type=str, default="models/trained_model", help="Where to save models")
    parser.add_argument("--results_dir", type=str, default="results", help="Where to save plots and logs")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    main(args)

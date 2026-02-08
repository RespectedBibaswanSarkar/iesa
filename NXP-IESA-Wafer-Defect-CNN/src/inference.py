import os
import argparse
import tensorflow as tf
import numpy as np

def load_trained_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return tf.keras.models.load_model(model_path)

def predict_interactive(model, class_names):
    print("\n\nStarting interactive prediction mode.")
    print("Enter the path to an image file or 'quit' to exit.")

    while True:
        image_path = input("Enter image path: ")
        if image_path.lower() == 'quit':
            break

        if not os.path.exists(image_path):
            print(f"Error: File not found at '{image_path}'.")
            continue

        try:
            img = tf.keras.utils.load_img(
                image_path,
                color_mode="grayscale",
                target_size=(256, 256)
            )
            arr = tf.keras.utils.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            pred = model.predict(arr, verbose=0)
            predicted_idx = np.argmax(pred)
            predicted_class = class_names[predicted_idx] if class_names else str(predicted_idx)
            confidence = np.max(pred) * 100

            print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")
        except Exception as e:
            print(f"Error processing image: {e}")

def main(args):
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model = load_trained_model(args.model_path)
    
    # We need class names. Ideally saved with model or config.
    # For now, we assume standard classes or try to infer if possible (hard with just .h5)
    # Let's use the standard list as default
    class_names = ["Bridge", "CMP", "Cracks", "Opens", "LER", "Vias"]
    
    if args.interactive:
        predict_interactive(model, class_names)
    elif args.input_path:
        if os.path.isdir(args.input_path):
            print(f"Processing directory {args.input_path}...")
            # directory logic
        else:
            # Single file
            img = tf.keras.utils.load_img(args.input_path, color_mode="grayscale", target_size=(256, 256))
            arr = tf.keras.utils.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr, verbose=0)
            cls = class_names[np.argmax(pred)]
            print(f"Result: {cls} ({np.max(pred)*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/trained_model/semiconductor_defect_cnn.h5")
    parser.add_argument("--input_path", type=str, help="Path to single image or dir")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    
    args = parser.parse_args()
    if not args.input_path and not args.interactive:
        # Default to interactive if nothing specified
        args.interactive = True
    main(args)

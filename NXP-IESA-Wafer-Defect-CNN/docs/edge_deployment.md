# Edge Deployment Strategy

## Overview
Deploying deep learning models on semiconductor inspection equipment requires low latency. We target **NXP i.MX 8M Plus** processors using TensorFlow Lite.

## Conversion Pipeline

### 1. Train in TensorFlow
- Train the model using `src/train.py`.
- The model is saved as `semiconductor_defect_cnn.h5`.

### 2. Convert to TensorFlow Lite
TensorFlow Lite is the native format for edge deployment with TensorFlow models.

```python
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('models/trained_model/semiconductor_defect_cnn.h5')

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save
with open('models/edge_ready_model/model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Optimization Techniques
- **Post-Training Quantization**: Reduces model size by ~4x (Float32 -> Int8) and improves inference speed on NPUs.
- **Pruning**: Removing sparse weights (requires re-training).

## Running on Edge
For NXP i.MX devices, use the `tflite_runtime` interpreter:
```python
import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Set input, invoke, get output...
```

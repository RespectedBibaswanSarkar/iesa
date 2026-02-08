# CNN-Based Classification of Semiconductor Wafer Defects Using SEM Images

**IESA–NXP DeepTech Hackathon Project**

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.10%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 1. Problem Statement
Semiconductor manufacturing is a highly complex process where even microscopic defects can lead to chip failure, reducing yield and increasing costs. Manual inspection of Scanning Electron Microscope (SEM) images is slow, error-prone, and unscalable. As feature sizes shrink to nanometer scales, automated, high-precision defect classification is critical for maintaining fabrication yield.

## 2. Motivation
This project aims to automate wafer inspection using Deep Learning. By deploying a lightweight Convolutional Neural Network (CNN) capable of identifying specific defect types (Shorts, Opens, Cracks, etc.) from SEM images, we can:
- **Reduce Inspection Time**: From minutes to milliseconds per image.
- **Increase Accuracy**: Eliminate human fatigue and variability.
- **Enable Edge Deployment**: Run inference directly on inspection tools using optimized edge models.

## 3. Defect Classes
The model classifies SIX specific defect types based on SEM morphology:
1. **Bridge**: Unwanted short circuits between interconnects.
2. **CMP (Chemical Mechanical Polishing)**: Surface scratches or uneven polishing artifacts.
3. **Cracks**: Stress-induced fractures in the wafer or die.
4. **Opens**: Broken interconnects causing open circuits.
5. **LER (Line Edge Roughness)**: Deviations in line width homogeneity affecting performance.
6. **Vias**: Missing or malformed vertical interconnect access points.

## 4. Repository Structure
```
NXP-IESA-Wafer-Defect-CNN/
│
├── dataset/                # Generated synthetic SEM images (6 classes)
├── src/                    # Source code (TensorFlow/Keras)
│   ├── model.py            # CNN Architecture
│   ├── train.py            # Training pipeline
│   ├── inference.py        # Prediction script
│   └── utils.py            # Data loading & helper functions
├── notebooks/              # Jupyter notebooks for analysis
├── edge_deployment/        # TFLite/ONNX conversion & benchmarks
├── results/                # Evaluation plots (Accuracy, Confusion Matrix)
├── docs/                   # Detailed documentation
└── models/                 # Model checkpoints
```

## 5. Methodology
### Data Generation
Due to the proprietary nature of real SEM data, we use a physics-based synthetic data generator (`generate_synthetic_data.py`).
*Note: The training script supports loading data from a zip file (`dataset_v2.zip`) for convenience.*

### Model Architecture
We utilize a custom lightweight CNN (`src/model.py`) implemented in **TensorFlow/Keras**:
- **Input**: 256x256 Grayscale
- **Blocks**: 3 Convolutional blocks (Conv2D -> ReLU -> MaxPooling)
- **Head**: Flatten -> Dense (128) -> Dropout -> Output (6 classes)
- **Optimization**: Adam, Sparse Categorical Crossentropy, Dropout for regularization

## 6. Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow, NumPy, Pandas, OpenCV
- See `requirements.txt`

### Installation
```bash
git clone https://github.com/yourusername/NXP-IESA-Wafer-Defect-CNN.git
cd NXP-IESA-Wafer-Defect-CNN
pip install -r requirements.txt
```

### Usage
**Step 1: Generate Dataset**
```bash
python generate_synthetic_data.py
# Optionally, zip the dataset if needed for portability
# python -m zipfile -c dataset_v2.zip dataset/
```

**Step 2: Train Model**
```bash
python src/train.py --epochs 10 --batch_size 16 --zip_path dataset_v2.zip
```
*Note: If `dataset_v2.zip` is not present, ensure the `dataset/` directory is populated using Step 1.*

**Step 3: Run Inference**
```bash
# Interactive Mode (Default)
python src/inference.py

# Single Image
python src/inference.py --input_path dataset/Bridge/bridge_0001.png
```

## 7. Results
Evaluation metrics on the validation set:
- **Accuracy**: >90% (simulated)
- **Inference Latency**: optimized for execution.

See `results/confusion_matrix.png` for detailed class-wise performance.

## 8. Edge Deployment
The Keras model (`.h5`) can be natively converted to **TensorFlow Lite** for deployment on edge devices like the Coral Dev Board or NXP i.MX benchmarks.
See `docs/edge_deployment.md` for conversion steps.

## 9. Future Scope
- Integration with real factory SEM logs.
- Real-time video stream processing.
- Deployment on NXP eIQ ML Software Environment.

---
**Developed for the IESA–NXP DeepTech Hackathon**

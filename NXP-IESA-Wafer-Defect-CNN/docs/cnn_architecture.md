# CNN Architecture: WaferDefectNet

## Overview
WaferDefectNet is a custom Convolutional Neural Network implemented in TensorFlow/Keras, designed properly for the texture-rich, high-contrast nature of SEM images.

## Architecture Diagram
```mermaid
graph TD
    Input[Input 256x256x1] --> RS[Rescaling 1./255]
    RS --> CV1[Conv2D 3x3, 32]
    CV1 --> RL1[ReLU]
    RL1 --> MP1[MaxPool 2x2]
    
    MP1 --> CV2[Conv2D 3x3, 64]
    CV2 --> RL2[ReLU]
    RL2 --> MP2[MaxPool 2x2]
    
    MP2 --> CV3[Conv2D 3x3, 128]
    CV3 --> RL3[ReLU]
    RL3 --> MP3[MaxPool 2x2]
    
    MP3 --> FL[Flatten]
    FL --> FC1[Dense 128]
    FC1 --> RL4[ReLU]
    RL4 --> DO[Dropout 0.5]
    DO --> FC2[Dense 6]
    FC2 --> SM[Softmax]
```

## Layer Specification
1. **Rescaling**: Normalizes input pixels to [0, 1].
2. **Conv Block 1**: 32 filters, 3x3 kernel, ReLU activation. Captures low-level edges.
3. **Conv Block 2**: 64 filters, 3x3 kernel, ReLU activation. Captures textures.
4. **Conv Block 3**: 128 filters, 3x3 kernel, ReLU activation. Captures complex shapes (vias, cracks).
5. **Dense Head**: 
   - Flatten 2D features to 1D vector.
   - Dense hidden layer (128 units).
   - Dropout (0.5) for regularization.
   - Output layer (6 units) with Softmax for probability distribution.

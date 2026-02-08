# Dataset Description

The dataset simulates high-magnification Scanning Electron Microscope (SEM) images of semiconductor wafers. It contains 6 balanced classes of common fabrication defects.

## Classes

### 1. Bridge
- **Description**: Unintended electrical connections between two conductive lines.
- **Impact**: Short circuits, device failure.
- **Visuals**: Horizontal or diagonal lines connecting vertical interconnects.

### 2. CMP (Chemical Mechanical Polishing) Defects
- **Description**: Scratches, pits, or uneven polishing residues.
- **Impact**: Layer non-uniformity, focus depth issues in subsequent lithography.
- **Visuals**: Random scratches, dark spots, or uneven background gradients.

### 3. Cracks
- **Description**: Mechanical fractures in the substrate or dielectric layers.
- **Impact**: Catastrophic structural failure.
- **Visuals**: Thin, jagged, dark lines propagating through the image.

### 4. Opens
- **Description**: Discontinuities in conductive lines (broken circuits).
- **Impact**: Open circuits, signal loss.
- **Visuals**: Gaps in vertical lines, often with irregular edges.

### 5. LER (Line Edge Roughness)
- **Description**: Excessive waviness in line edges.
- **Impact**: Signal integrity issues, leakage variability.
- **Visuals**: Vertically oriented lines with high-frequency edge wiggles.

### 6. Vias
- **Description**: Vertical Interconnect Access points (holes).
- **Impact**: Critical for layer-to-layer connection; malformed vias cause opens/shorts.
- **Visuals**: Circular features with distinct rim contrast.

## Specifications
- **Format**: PNG (Grayscale)
- **Resolution**: 256 x 256 pixels
- **Split**: 80% Training, 20% Validation
- **Source**: Synthetic Generation (Physics-Informed)

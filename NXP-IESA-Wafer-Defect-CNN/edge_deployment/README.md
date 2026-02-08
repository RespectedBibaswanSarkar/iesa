# Edge Deployment Resources

This directory contains resources for deploying the trained model to edge devices.

## Contents

- **conversion_scripts/**: (Placeholder) Python scripts for TFLite/ONNX conversion. see `docs/edge_deployment.md` for manual instructions.
- **benchmarks/**: (Placeholder) Results from `benchmark_model.py` runs on target hardware.

## Quick Start (ONNX)

```bash
python -m src.utils.export_onnx --model_path models/trained_model/best_model.pth --output models/edge_ready_model/model.onnx
```

*Note: Ensure you have `onnx` and `onnxruntime` installed.*

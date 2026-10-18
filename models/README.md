# DeepHeartSeg Legacy Weights & ONNX Models

This directory stores the pretrained DeepHeartSeg TensorFlow 1.14 / Keras model weights and their converted ONNX models:

1. `deepheartseg_step1.hdf5` & `deepheartseg_loc.onnx`:
   - Step 1 Cardiac Localization model (`64 x 128 x 128` input).
   - Predicts the 3D cardiac bounding box from whole-chest CT.

2. `deepheartseg_step2.hdf5` & `deepheartseg_heart.onnx`:
   - Step 2 Cardiac Chamber & Pericardium Segmentation model (`112 x 128 x 128` input).
   - Segment myocardium and pericardium within the cropped cardiac volume.

### Source
Weights originate from the Harvard AIM DeepHeartSeg repository:
`https://github.com/AIM-Harvard/DeepHeartSeg` (releases / model assets: `GPU_1.hdf5`).

### ONNX Conversion
Converted to ONNX with transposed kernel dimensions `(out_c, in_c, D, H, W)` and validated for bit-level / numerical equivalence against the original TensorFlow 1.14 inference pipeline.

# M2 Task 1: Build and Train the Baseline Model

The baseline is a 1-D CNN trained on the standardized bearing-signal training
split. The model definition and training workflow are in
[`M2_01_baseline_model.m`](../../notebooks/M2_01_baseline_model.m); the trained
network is saved as `data/models/baseline/baseline_net.mat`.

The CNN uses three convolutional blocks with batch normalization and ReLU,
max-pooling after the first two blocks, global average pooling, dropout, and a
three-class output layer. Training uses Adam for up to 30 epochs and validates
on the validation split during training. Task 2 records the baseline accuracy,
per-class metrics, checkpoint size, and inference timing.

Run the MATLAB file from the repository root after the standardized data and
training data files are available:

```matlab
run("notebooks/M2_01_baseline_model.m")
```

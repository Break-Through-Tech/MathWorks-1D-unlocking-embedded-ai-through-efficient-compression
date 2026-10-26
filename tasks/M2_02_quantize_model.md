# M2 Task 2: Quantize the baseline model

The trained 1-D CNN checkpoint is `data/baseline_net.mat`, originally created on the `m2-01-baseline-model` branch. Its training notebook is [`M2_01_baseline_model.mlx`](../notebooks/M2_01_baseline_model.mlx). The fitted scaler is `data/scaler.mat`.

Run [`M2_02_quantize_model.m`](../scripts/M2_02_quantize_model.m) in MATLAB R2024b or newer from the repository root:

```matlab
run("scripts/M2_02_quantize_model.m")
```

This requires Deep Learning Toolbox, the Deep Learning Toolbox Model Compression Library add-on, and Fixed-Point Designer. The script prepares the saved `dlnetwork`, calibrates INT8 weights and activations with **training data only**, then quantizes it for MATLAB simulation. It applies the saved training scaler to validation signals and prints baseline and quantized validation accuracy. A successful run writes `data/quantized_net.mat`.

The script checks `quantizationDetails(qNet).IsQuantized` before saving. MATLAB simulation uses fixed-point arithmetic; the saved MAT-file size is also printed, but it is not a deployment memory measurement. The test split remains reserved for the final model comparison.

This environment has no MATLAB installation, so the quantized checkpoint and numerical results could not be generated or verified here. Run the script in a licensed MATLAB environment before reporting accuracy or compression gains.

References: [MathWorks quantization workflow](https://www.mathworks.com/help/deeplearning/ug/quantization-of-deep-neural-networks.html), [supported quantization layers](https://www.mathworks.com/help/deeplearning/ug/supported-layers-for-quantization.html).

# M2 Task 5: Apply Model Quantization

The post-training quantization script calibrates the baseline network on
standardized training signals, converts its learnables to INT8, validates the
quantized network, and saves it to `data/quantized_net.mat`.

Run from the repository root in MATLAB. The script requires MATLAB R2024b or
later, Deep Learning Toolbox, the Model Compression Library add-on, and
Fixed-Point Designer:

```matlab
run("scripts/M2_02_quantize_model.m")
```

The comparison script for Task 6 expects this saved checkpoint and reports
validation accuracy, per-class F1, checkpoint size, and single-sample inference
latency alongside the baseline, pruned, and projected models.

The separate `M2_01_quantize_data.md` write-up documents INT8 quantization of
the input signals; quantizing input data alone does not quantize model weights.

# M2 Task 5: Apply Model Quantization

The post-training quantization script calibrates the baseline network on
standardized training signals, converts its learnables to INT8, validates the
quantized network, and saves it to `data/models/quantized/quantized_net.mat`.

Run from the repository root in MATLAB. The script requires MATLAB R2024b or
later, Deep Learning Toolbox, the Model Compression Library add-on, and
Fixed-Point Designer:

```matlab
run("scripts/M2_02_quantize_model.m")
```

The comparison script for Task 6 expects this saved checkpoint and reports
validation accuracy, per-class F1, checkpoint size, and single-sample inference
latency alongside the baseline, pruned, and projected models.

To report the quantized model's size and speed alongside the baseline without
running the other models, use:

```matlab
addpath("scripts"); results = M2_05_benchmark_quantized();
```

The benchmark prints checkpoint size, estimated parameter memory, and median
CPU milliseconds per sample after 200 warmups. It uses the same standardized
validation signal for both models. Its timing measures MATLAB fixed-point
simulation, not deployed INT8 execution.

## Input Data Quantization

Separately from the model, the input signals can also be stored as INT8.
Quantizing input data alone does not quantize model weights.

Run `python scripts/quantize_data.py` to regenerate `data/quantized.mat` from
the raw and standardized datasets. The script uses the training-column mean
and standard deviation already used by `standardized.mat` to transform the
validation split. It then computes one symmetric int8 scale from the largest
absolute standardized training value and applies it to every split. Labels,
scale, zero point, and training statistics are stored with the signals.

| Split | Signals | Clipped values | Mean absolute reconstruction error |
|-------|---------|----------------|------------------------------------|
| Train | 393 x 5,000 | 0 / 1,965,000 | 0.035197 |
| Validation | 27 x 5,000 | 10 / 135,000 | 0.035271 |
| Test | 102 x 5,000 | 80 / 510,000 | 0.035501 |

`quantized.mat` is about 2.7 MB, compared with about 19 MB for
`standardized.mat`. The quantized file also includes validation signals and
training statistics, so this is a file-size comparison rather than an exact
like-for-like compression ratio. Dequantize with `X_int8 * scale` before using
models that expect floating-point inputs. Input quantization alone does not
reduce model weight size or change its arithmetic.

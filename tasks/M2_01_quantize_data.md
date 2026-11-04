# Task 1: Quantize Data

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

# M2 Task 6: Compare Compression Results

## Goal

Compare the baseline 1-D CNN with the Taylor-pruned, projected, and INT8
quantized models using the same validation split and inference timing method.

## Checkpoints

The following checkpoints are included in this branch:

```matlab
run("notebooks/M2_03_pruning.m")
run("notebooks/M2_04_projection.m")
run("scripts/M2_02_quantize_model.m")
```

The comparison expects these files:

| Model | Checkpoint |
|---|---|
| Baseline | `data/models/baseline/baseline_net.mat` |
| Taylor-pruned | `data/models/pruned/taylor_pruned_net.mat` |
| Projected + fine-tuned | `data/models/projected/projected_finetuned_net.mat` |
| Quantized INT8 | `data/models/quantized/quantized_net.mat` |

To regenerate a checkpoint, run its task script first:

```matlab
run("notebooks/M2_03_pruning.m")
run("notebooks/M2_04_projection.m")
run("scripts/M2_02_quantize_model.m")
```

## Run

```matlab
run("notebooks/M2_06_compare_compression.m")
```

The script evaluates accuracy and per-class F1 on the validation split,
measures checkpoint file size, and reports single-sample inference latency.
Timing runs every model on the CPU, with 200 warmups followed by the median of
100 single-sample predictions. Run the models on the same machine for a
meaningful latency comparison. The CSV labels this value
`CpuInferenceMsPerSample`. Preprocessing and file loading are excluded.
Checkpoint size is the saved `.mat` file size, including MATLAB serialization
overhead. `ParameterMemoryMB` estimates memory for learnable parameters using
`estimateNetworkMetrics` for every model.

## Existing comparison results

The checked-in CSV was generated before the CPU timing change. Its
`AutoInferenceMsPerSample` values used MATLAB's automatic execution environment.
MATLAB can choose a GPU for floating-point networks but runs this quantized
network on the CPU, so those timings are not a controlled speed comparison.
The old script also estimated floating-point parameter memory by counting
learnables, while it used `estimateNetworkMetrics` for the quantized model;
rerun to put every model on the same memory metric.
The reported 19.925 ms/sample for the quantized model is a measurement of
MATLAB fixed-point simulation on that run, not an optimized INT8 deployment
benchmark. Rerun the script to obtain comparable CPU timings and replace the
checked-in CSV, MAT table, and plot.

The existing quantized checkpoint is 2.213187 MB, while its estimated
parameter memory is 0.047974 MB. The projected and fine-tuned model has the
smallest values in the current CSV: 0.047876 MB checkpoint size and 0.040424 MB
estimated parameter memory. Quantization reduced parameter memory relative to
the baseline, but the MATLAB checkpoint is larger because of serialization
overhead.

The script writes the comparison and per-class F1 tables as CSV and MAT files,
and creates `M2_06_comparison.png` under `tasks/milestone_2/figures/`.
Generated checkpoints and comparison outputs are not committed by the script.

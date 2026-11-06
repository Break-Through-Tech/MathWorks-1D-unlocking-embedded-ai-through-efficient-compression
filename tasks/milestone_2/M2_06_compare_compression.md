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
Timing uses 200 warmups followed by the median of 100 predictions for each
model. Run the models on the same machine for a meaningful latency comparison.
Checkpoint size is the saved `.mat` file size, including MATLAB serialization
overhead; it is not the same as runtime parameter memory.

The script writes the comparison and per-class F1 tables as CSV and MAT files,
and creates `M2_06_comparison.png` under `tasks/milestone_2/figures/`.
Generated checkpoints and comparison outputs are not committed by the script.

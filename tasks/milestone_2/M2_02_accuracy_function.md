# M2 Task 2: Accuracy Function

**Owner:** Sahana  
**Milestone:** October - Model Development & Compression  
**Issue:** [#15](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/15)

## Goal

Evaluate baseline performance before any compression. Get accuracy, F1 score per class, and note size (MB) and inference speed (ms/sample). This is the comparison point for the compression tasks (pruning, projection, quantization).

## Files

| File | What it does |
|------|--------------|
| [`M2_02_evaluate_baseline.m`](../../scripts/M2_02_evaluate_baseline.m) | Loads `data/baseline_net.mat`, standardizes the validation set with the saved training scaler, and reports the baseline metrics. |
| [`M2_02_evaluateModel.m`](../../notebooks/M2_02_evaluateModel.m) | Reusable function that takes any trained network plus test data and returns the same kind of metrics. Use this to evaluate the compressed models in Tasks 3–5. |

## How to Run

From the repository root in MATLAB (Or in Cursor terminal):

```matlab
% Baseline metrics
addpath("scripts");
metrics = M2_02_evaluate_baseline();

% Any other model (e.g. a pruned or quantized network)
addpath("notebooks");
results = M2_02_evaluateModel(net, XTest, YTest, classNames, modelFilePath);
```

## Metrics Reported

- **Accuracy:** fraction of validation samples classified correctly.
- **Per-class precision, recall, and F1**, plus macro F1 (the average F1 across classes).
- **Confusion matrix.**
- **Model size (MB):** size of the saved `.mat` checkpoint on disk.
- **Inference speed (ms/sample).**

`M2_02_evaluateModel` also reports per-class TPR, FPR, specificity, and the number of learnable parameters.

## Notes

- Metrics use the **validation** split. The test split is kept for the final comparison between models.
- The two files measure speed differently. `M2_02_evaluate_baseline` times one sample at a time (median of 100 runs, after 200 untimed warm-up predictions). `M2_02_evaluateModel` times one batch prediction over the whole set and divides by the number of samples. Batching is faster per sample, so only compare speeds measured with the same function.
- Model size is the checkpoint file size, not the memory the model would use on an embedded device. MATLAB compresses `.mat` files, so a model full of zeros (e.g. after magnitude pruning) has a smaller file without being any smaller in memory.

## Baseline Results

Produced by `M2_02_evaluate_baseline` on the validation split (27 samples, 9 per class). Run on 2026-10-28 with MATLAB R2025b on a macOS laptop (CPU).

| Metric | Value |
|--------|-------|
| Accuracy | 1.0000 (27 / 27) |
| Macro F1 | 1.0000 |
| Checkpoint size | 0.1959 MB |
| Inference speed | ~4.4 ms/sample (4.16–4.64 ms across four runs; median of 100 single-sample runs) |

**Per-class metrics**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| InnerRaceFault | 1.00 | 1.00 | 1.00 |
| Normal | 1.00 | 1.00 | 1.00 |
| OuterRaceFault | 1.00 | 1.00 | 1.00 |

**Confusion matrix** (rows = true class, columns = predicted class)

| | InnerRaceFault | Normal | OuterRaceFault |
|---|---|---|---|
| **InnerRaceFault** | 9 | 0 | 0 |
| **Normal** | 0 | 9 | 0 |
| **OuterRaceFault** | 0 | 0 | 9 |

**Takeaways for compression**

- The validation set is small (27 samples), so each misclassification costs about 3.7 accuracy points. Expect accuracy to move in coarse steps when comparing compressed models.
- Inference time depends on the machine and includes `minibatchpredict` call overhead, which dominates for a single small sample. Compare compressed models on the same machine with the same timing method.
- An earlier run recorded 18.83 ms/sample. That number was inflated: the first ~100 predictions in a MATLAB session run at 17–19 ms, and only after that does speed settle. The script now runs 200 untimed warm-up predictions before timing.
- Even after warm-up, single-sample timing varies between about 3 and 6 ms from run to run, so speed differences under ~2 ms are noise.

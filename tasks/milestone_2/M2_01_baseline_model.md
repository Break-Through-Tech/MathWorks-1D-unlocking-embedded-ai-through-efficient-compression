# M2 Task 1: Build and Train the Baseline Model

**Owner:** Bez  
**Milestone:** October - Model Development & Compression  
**Issue:** [#[number]]([issue link])

## Goal

Build and train a small 1-D CNN on the standardized bearing-signal data, and track accuracy and loss on the training and validation sets. This is the baseline that the compression tasks (pruning, projection, quantization) start from and are compared against.

## Files

| File | What it does |
|------|--------------|
| [`M2_01_baseline_model.m`](../../notebooks/M2_01_baseline_model.m) | Loads the standardized data, standardizes the validation set with the training scaler, defines the CNN, trains it, and saves the network. |
| `data/models/baseline/baseline_net.mat` | The trained network (`net`) and its test accuracy (`testAcc`). Input for Tasks 2-5. |
| `[path]/scaler.mat` | The training mean (`mu`) and standard deviation (`sigma`), one value per signal position (1 x 5000). Needed to standardize any new data the same way. |

## How to Run

From the repository root in MATLAB:

```matlab
run("notebooks/M2_01_baseline_model.m")
```

The standardized data file (`standardized.mat`) and the raw `train.mat` and `val.mat` must be on the MATLAB path. The script needs MATLAB R2023b or newer for `trainnet`.

## Model

The CNN uses three convolutional blocks, each with batch normalization, with max-pooling after the first two blocks. After the blocks, global average pooling, dropout (0.3), and a fully connected layer with a softmax output give one score per class.

| Block | Filters | Filter length | Signal length after |
|-------|---------|---------------|---------------------|
| 1 | 16 | 64 | 1,250 (after pooling) |
| 2 | 32 | 32 | 312 (after pooling) |
| 3 | 64 | 16 | 312 (no pooling) |

- 16 layers and **50,707 learnable parameters**.
- Input: one channel, 5,000 time steps per window.
- Output: three classes (InnerRaceFault, Normal, OuterRaceFault).

## Training Setup

- Optimizer: Adam, initial learning rate 0.001.
- Up to 30 epochs, mini-batch size 32, data reshuffled every epoch.
- Loss: cross-entropy.
- Validation data is checked about once per epoch (every 13 iterations) and the best validation network is kept.
- Hardware: single CPU. Training took about 40 seconds.

## Data Handling

- `standardized.mat` only contains the standardized train and test sets, so the validation set was standardized separately from the raw `val.mat`.
- The mean and standard deviation come from the **training set only**, one value per column, and are applied to the validation set. Nothing is fit on validation or test data.
- As a check, recomputing the standardized training data in MATLAB matched the September data to within about 7e-15, so it uses the same per-column method.
- Each window is stored as a 5,000 x 1 column (time steps x channel) inside a cell array. Storing it as a 1 x 5,000 row gives a "channel dimension" error in `trainnet`.

## Results

Run on 2026-10-22 in MATLAB Online.

| Metric | Value |
|--------|-------|
| Training accuracy | 100% |
| Validation accuracy | 100% (27 / 27) |
| Test accuracy | 100% (102 / 102) |
| Learnable parameters | 50,707 |
| Saved checkpoint size | about 0.196 MB |

Training and validation loss both fall to near zero by about iteration 50 (around epoch 4) and stay flat for the rest of the 30 epochs, and the two curves stay together.

## Notes

- Perfect accuracy most likely means this dataset is easy for the model and the data were were given is extremely clean.
- The validation set is small (27 samples), so each sample is worth about 3.7 accuracy points.
- Per-class metrics, model size, and inference speed are covered in Task 2.

## Final Note

The baseline is trained and saved. The next step is to evaluate it in detail (Task 2), then compress it (Tasks 3-5).
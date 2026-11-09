# Smarter Devices, Smaller Models: Unlocking Embedded AI Through Efficient Compression

This Fall 2026 Break Through Tech AI Studio project, developed with MathWorks, explores whether a neural network can identify rolling-element bearing faults while using fewer model parameters. We trained a one-dimensional convolutional neural network (1-D CNN) on vibration signals, compared pruning, projection, and quantization, and generated C code for the projected model.

## Team

| Name | GitHub handle | Contribution |
|---|---|---|
| Bezawit Gessesse | @bezqwit | Project coordination, data exploration, and baseline model |
| Chikaosolu Nnadozie | @cnnadozi | Magnitude and Taylor pruning; performance comparison |
| Anisha Bollapragada | @anisharaob | Data preprocessing and standardization |
| Lena Ngo | @lenan14 | Initial data analysis, compression research, and MATLAB C code generation and prediction verification |
| Huy Huynh | @baohuy1303 | Data loading and validation, and model compression with projection |

## Project Overview

The challenge was to build a fault-diagnosis model for vibration signals and study the trade-offs between accuracy, model size, and inference time that matter for embedded use. The project uses the Rolling Element Bearing Fault Diagnosis dataset, with three labels: Normal, InnerRaceFault, and OuterRaceFault. The supplied signals were resampled to 48,828 Hz and split into 5,000-sample windows.

Our baseline is a 1-D CNN trained on standardized signal windows. We evaluated magnitude pruning and Taylor filter pruning, low-rank projection, and post-training INT8 quantization. As a stretch goal, we used MATLAB Coder to generate C for the projected, fine-tuned network and checked its predictions against MATLAB on the held-out test split.

## Highlights

- On the 27-sample validation split, the baseline and the Taylor-pruned, projected, and quantized models each achieved 100% accuracy and macro F1 in the recorded comparison.
- Projection reduced the baseline from 50,707 to 10,106 learnable parameters (about 80% fewer) while retaining 100% validation accuracy. Fine-tuning kept the same validation score.
- MATLAB Coder generated C for the projected, fine-tuned model. Its MEX predictions matched MATLAB's predicted classes on all 102 held-out test signals; both classified all 102 correctly. The largest score difference was `6.11e-10`.
- The generated code was verified in MATLAB Online.

## Repository Guide

| Path | Contents |
|---|---|
| [`data/`](data/) | Dataset splits, preprocessing outputs, and saved model checkpoints. See [`data/README.md`](data/README.md) for the dataset source, license, and file details. |
| [`notebooks/`](notebooks/) | Data exploration notebooks and MATLAB scripts for baseline training, pruning, projection, and comparison. |
| [`scripts/`](scripts/) | Quantization and embedded code-generation scripts. |
| [`tasks/`](tasks/) | Milestone task index, write-ups, and comparison figures. Start with [`tasks/TASKS.md`](tasks/TASKS.md). |
| [`Challenge-Project-Overview.md`](Challenge-Project-Overview.md) | The original challenge description and success criteria. |

## Setup and running the project

The data and trained checkpoints are included in `data/`. The modeling and compression workflows use MATLAB. From the repository root in MATLAB, run:

```matlab
run("notebooks/M2_01_baseline_model.m")       % Train the baseline
run("notebooks/M2_03_pruning.m")              % Magnitude and Taylor pruning
run("notebooks/M2_04_projection.m")           % Projection sweep and fine-tuning
run("scripts/M2_02_quantize_model.m")          % Post-training model quantization
run("notebooks/M2_06_compare_compression.m")  % Compare saved checkpoints
```

The comparison script expects the corresponding checkpoints to exist under `data/models/`; run the training or compression script first if a checkpoint is missing. The MATLAB workflows require Deep Learning Toolbox. Projection and quantization also require the Model Compression Library support package; quantization additionally requires Fixed-Point Designer and MATLAB R2024b or later. The baseline training script documents MATLAB R2023b or later. MATLAB Coder is needed for the code-generation stretch goal.

Python is used for the separate input-signal quantization script. To set up that part from the repository root:

```powershell
python -m pip install -r requirements.txt
python scripts/quantize_data.py
```

This generates `data/quantized.mat`; it quantizes the input data, not the model weights. Instructions and details for generating embedded code are in [`tasks/milestone_3/M3_01_embedded_code_generation.md`](tasks/milestone_3/M3_01_embedded_code_generation.md).

## Data exploration and preparation

The dataset contains 393 training windows (131 per class), 27 validation windows, and 102 test windows. Each window has 5,000 samples. The team checked the matrix shapes, labels, class counts, and values for missing or invalid entries. Unusually high or low signal values were retained because they may reflect actual fault behavior.

Signals are standardized per sample position using the mean and standard deviation calculated from the training split. Those same training statistics are applied to validation and test data; they are saved in `data/scaler.mat`. The preprocessed files and MATLAB/Python loading examples are documented in [`data/README.md`](data/README.md).

## Model and evaluation

The baseline is a three-block 1-D CNN with 16, 32, and 64 convolutional filters, followed by global average pooling, dropout, and a three-class output layer. It has 50,707 learnable parameters. Training uses Adam and cross-entropy loss; the baseline configuration and training notes are in [`tasks/milestone_2/M2_01_baseline_model.md`](tasks/milestone_2/M2_01_baseline_model.md).

The table below reports the saved comparison results. Accuracy and macro F1 are from the 27-sample validation split. Checkpoint size is the saved MATLAB `.mat` file size; parameter memory is an estimate based on stored parameters, not a measurement on an embedded target. Inference times are specific to the machine and MATLAB run used for the comparison.

| Model | Validation accuracy | Macro F1 | Checkpoint size | Parameter memory | Inference time |
|---|---:|---:|---:|---:|---:|
| Baseline | 100% | 1.00 | 0.196 MB | 0.203 MB | 2.53 ms/sample |
| Taylor-pruned | 100% | 1.00 | 0.073 MB | 0.072 MB | 2.65 ms/sample |
| Projected + fine-tuned | 100% | 1.00 | 0.048 MB | 0.040 MB | 2.98 ms/sample |
| Quantized INT8 | 100% | 1.00 | 2.213 MB | 0.048 MB | 19.93 ms/sample |

The comparison script uses 200 warm-up predictions and the median of 100 single-sample predictions. These measurements include MATLAB prediction-call overhead and should only be compared within the same run environment. In this comparison, quantization reduced parameter memory but did not make inference faster, and its saved checkpoint was larger than the other checkpoints. Magnitude pruning also creates zero weights but does not reduce the dense network's parameter count or guarantee faster inference.

Separately, storing the standardized signal data as INT8 reduced the file from about 19 MB to about 2.7 MB. This is not a like-for-like model-size comparison: the quantized data file also contains the validation split and training statistics.

## Limitations and next steps

The validation set is small, so a perfect score is encouraging but not enough to establish how the model will perform on new bearings, operating conditions, or measurement hardware. The dataset was collected on one test rig under controlled laboratory conditions. Inference timing here is from MATLAB on a computer, not an embedded board.

A useful next step would be testing on data from additional rigs or operating conditions, then running the generated C code on a specified target device and measuring its inference speed and memory use.

## References and acknowledgements

- [MathWorks rolling-element bearing fault diagnosis example](https://www.mathworks.com/help/predmaint/ug/rolling-element-bearing-fault-diagnosis-using-deep-learning.html)
- [Original bearing dataset source](https://data-acoustics.com/measurements/bearing-faults/bearing-2). Dataset terms are described in [`data/README.md`](data/README.md).
- Many thanks to our Challenge Advisor Shanmukh Srinivas, AI Studio Coach Rashidah Carr, MathWorks, and Break Through Tech for supporting the project.
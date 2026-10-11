# M2 Task 8: Researching Model Compression

**Owner:** Lena  
**Milestone:** October - Model Development & Compression

## Goal

After a bit of digging, I created this document to give the team a starting point for trying pruning, projection, and quantization on our 1D CNN so we can skip the tedious work. We want to compare model accuracy, file size, and prediction time.

## Before We Start

The project overview mentions both MATLAB and Python, but the team still needs to choose which tools to use. However, we have made it clear to at least experiment with MATLAB since it can give us an edge compared to only using Python.

“Projection” can mean different things. One option is **low-rank factorization**, which replaces a large layer with two smaller ones. This could make the model smaller, but might also affect accuracy or speed. Our team should check with the advisor before using this as the definition for Task #4.

## Compression Methods

| Method | Possible first step | What to keep in mind |
|---|---|---|
| **Pruning** | Remove some of the model's smaller weights, then check its performance. Fine-tune it if the framework supports that. | A model with more zero weights is not always smaller or faster when saved or run. Measure the actual file and prediction time. |
| **Projection** | If the team agrees, try low-rank factorization on a suitable layer. Compare a few settings using the validation data. | The smaller layers may reduce the number of weights, but could change accuracy and speed. |
| **Quantization** | Try converting weights from 32-bit to 8-bit, if the chosen tools support it. | This can reduce file size. Speed and accuracy depend on the tools and hardware. Use training data for calibration, not test data. |

## Keep the Comparison Fair

- Use the same baseline model and data preparation for every method.
- Choose settings with the validation set. Save the test set for the final comparison.
- Report accuracy, macro F1, and F1 for each fault class.
- Measure the saved model's file size and prediction time using the same computer and tools.

For prediction time, use one signal at a time and repeat the measurement. Report the median time and how many runs you used. Note whether the timing includes data preparation.

## Results to Fill In

| Model | Settings | Accuracy | Macro F1 | Normal F1 | Inner fault F1 | Outer fault F1 | File size (MB) | Time (ms/sample) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | - | | | | | | | |
| Pruned | | | | | | | | |
| Projected | | | | | | | | |
| Quantized | | | | | | | | |

## Questions for the Team

1. Does the advisor agree that projection means low-rank factorization?
2. What computer or device should we use to measure prediction time?

## References I Found or We Already Had

- [Challenge Project Overview](../Challenge-Project-Overview.md) - project goals, data, and suggested tools.
- PyTorch, [Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html).
- TensorFlow Model Optimization, [Post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training).
- [Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation](https://arxiv.org/abs/1404.0736) - background on low-rank methods that I mentioned at the beginning.

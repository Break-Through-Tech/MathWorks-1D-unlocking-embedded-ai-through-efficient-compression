# Tasks

Create a markdown file for each completed task as a write-up and store it under the corresponding milestone folder for organization.

All project tasks, taken from this repo's [GitHub Issues](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues).

Task write-ups and notebooks follow the naming convention in [AGENTS.md](../AGENTS.md#naming-convention).

## M1: September Milestone

### Task 1: Download Data
- **Issue:** [#7](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/7)
- **Assignees:** @baohuy1303
- **Description:** Download the dataset files and get them opening correctly in our notebook/workspace.
- **Write-up:** [M1_01_load_data.md](milestone_1/M1_01_load_data.md)

### Task 2: Initial Data Analysis
- **Issue:** [#2](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/2)
- **Assignees:** @lenan14
- **Description:** Do initial data analysis and address missing values and outliers. Use Python packages/functions to survey the data so we address missing values and outliers.
- **Write-up:** [M1_02_initial_data_analysis.md](milestone_1/M1_02_initial_data_analysis.md)

### Task 3: Equal Row Counts
- **Issue:** [#8](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/8)
- **Assignees:** @bezqwit
- **Description:** Count how many samples we have for each of the 3 categories (normal, outer race fault, inner race fault).
- **Write-up:** [M1_03_count_samples.md](milestone_1/M1_03_count_samples.md)

### Task 4: Visualize Signals
- **Issue:** [#3](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/3)
- **Assignees:** @sbalagali
- **Description:** Plot example signals from each category to visually check for any differences or things to account for later on.
- **Write-up:** [M1_04_plot_data.md](milestone_1/M1_04_plot_data.md)

### Task 5: Standardize Data
- **Issue:** [#4](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/4) (see also [#5](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/5): "Conduct exploratory data analysis & Standardize the set")
- **Assignees:** @AnishaRaoB, @Tobechukwu-Ikenwe
- **Description:** Standardize/fit the training set. Then transform the test set using the same transformation from the training set. Save the standardized data for model training.
- **Write-up:** [M1_05_standardize_data.md](milestone_1/M1_05_standardize_data.md)

### Task 6: Update the README + Other Docs
- **Issue:** [#6](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/6)
- **Assignees:** @sbalagali, @cnnadozi, @lenan14, @bezqwit, @baohuy1303, @Tobechukwu-Ikenwe
- **Description:** Add all updates and task completions to the README for documentation.
- **Write-up:** [M1_06_update_docs.md](milestone_1/M1_06_update_docs.md)

## M2: October Milestone

### Task 1: Build Baseline Model
- **Description:** Train a 1-D CNN on standardized bearing signals.
- **Implementation:** [M2_01_baseline_model.m](../notebooks/M2_01_baseline_model.m)
- **Write-up:** [M2_01_baseline_model.md](milestone_2/M2_01_baseline_model.md)

### Task 2: Evaluate Baseline Performance
- **Description:** Report accuracy, per-class F1, model size, and inference speed.
- **Write-up:** [M2_02_accuracy_function.md](milestone_2/M2_02_accuracy_function.md)

### Task 3: Apply Pruning
- **Description:** Compare magnitude and Taylor pruning against the baseline.
- **Implementation:** [M2_03_pruning.m](../notebooks/M2_03_pruning.m)
- **Write-up:** [M2_03_pruning.md](milestone_2/M2_03_pruning.md)

### Task 4: Apply Projection
- **Description:** Compress the baseline with projection and measure its performance.
- **Implementation:** [M2_04_projection.m](../notebooks/M2_04_projection.m)
- **Write-up:** [M2_04_projection.md](milestone_2/M2_04_projection.md)

### Task 5: Apply Quantization
- **Description:** Quantize the trained baseline model and measure its performance.
- **Implementation:** [M2_02_quantize_model.m](../scripts/M2_02_quantize_model.m)
- **Write-up:** [M2_05_quantization.md](milestone_2/M2_05_quantization.md)

### Task 6: Compare Compression Results
- **Description:** Compare baseline, pruned, projected, and quantized models and plot accuracy, size, speed, and per-class F1.
- **Implementation:** [M2_06_compare_compression.m](../notebooks/M2_06_compare_compression.m)
- **Write-up:** [M2_06_compare_compression.md](milestone_2/M2_06_compare_compression.md)

### Task 7: Research Model Compression
- **Issue:** [#20](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/20)
- **Description:** Research pruning, projection, and quantization and document key implementation considerations and trade-offs for the team.
- **Write-up:** [M2_07_compression_research.md](milestone_2/M2_07_compression_research.md)

## M3: Embedded Code Generation Stretch Goal

### Task 1: Generate Embedded Code
- **Description:** Generate C/C++ code for the projected, fine-tuned network and verify its predictions against MATLAB on the held-out test set.
- **Implementation:** [M3_01_generate_embedded_code.m](../scripts/M3_01_generate_embedded_code.m), [M3_01_embedded_predict.m](../scripts/M3_01_embedded_predict.m)
- **Write-up:** [M3_01_embedded_code_generation.md](milestone_3/M3_01_embedded_code_generation.md)
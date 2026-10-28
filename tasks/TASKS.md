# Tasks

All project tasks, taken from this repo's [GitHub Issues](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues).

Task write-ups and notebooks follow the naming convention in [AGENTS.md](../AGENTS.md#naming-convention).

## M1: September Milestone

### Task 1: Download Data
- **Issue:** [#7](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/7)
- **Assignees:** @baohuy1303
- **Description:** Download the dataset files and get them opening correctly in our notebook/workspace.
- **Write-up:** [M1_01_load_data.md](M1_01_load_data.md)

### Task 2: Initial Data Analysis
- **Issue:** [#2](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/2)
- **Assignees:** @lenan14
- **Description:** Do initial data analysis and address missing values and outliers. Use Python packages/functions to survey the data so we address missing values and outliers.
- **Write-up:** [M1_02_initial_data_analysis.md](M1_02_initial_data_analysis.md)

### Task 3: Equal Row Counts
- **Issue:** [#8](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/8)
- **Assignees:** @bezqwit
- **Description:** Count how many samples we have for each of the 3 categories (normal, outer race fault, inner race fault).
- **Write-up:** [M1_03_count_samples.md](M1_03_count_samples.md)

### Task 4: Visualize Signals
- **Issue:** [#3](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/3)
- **Assignees:** @sbalagali
- **Description:** Plot example signals from each category to visually check for any differences or things to account for later on.
- **Write-up:** [M1_04_plot_data.md](M1_04_plot_data.md)

### Task 5: Standardize Data
- **Issue:** [#4](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/4) (see also [#5](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/5): "Conduct exploratory data analysis & Standardize the set")
- **Assignees:** @AnishaRaoB, @Tobechukwu-Ikenwe
- **Description:** Standardize/fit the training set. Then transform the test set using the same transformation from the training set. Save the standardized data for model training.
- **Write-up:** [M1_05_standardize_data.md](M1_05_standardize_data.md)

### Task 6: Update the README + Other Docs
- **Issue:** [#6](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/6)
- **Assignees:** @sbalagali, @cnnadozi, @lenan14, @bezqwit, @baohuy1303, @Tobechukwu-Ikenwe
- **Description:** Add all updates and task completions to the README for documentation.
- **Write-up:** [M1_06_update_docs.md](M1_06_update_docs.md)

## M2: October Milestone

### Task 1: Build + Train Baseline Model
- **Issue:** [#14](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/14)
- **Description:** Train the 1-D CNN baseline and save its checkpoint and scaler.
- **Notebook:** [M2_01_baseline_model.mlx](../notebooks/M2_01_baseline_model.mlx)

### Task 2: Accuracy Function
- **Issue:** [#15](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/15)
- **Assignees:** @sbalagali
- **Description:** Evaluate baseline performance. Get accuracy, F1 score per class, and note size (MB) and inference speed (ms/sample) before any compression. This is the comparison point.
- **Code:** [M2_02_evaluate_baseline.m](../scripts/M2_02_evaluate_baseline.m), [M2_02_evaluateModel.m](../notebooks/M2_02_evaluateModel.m)
- **Write-up:** [M2_02_accuracy_function.md](milestone_2/M2_02_accuracy_function.md)

### Task 3: Apply Pruning
- **Issue:** [#16](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/16)
- **Assignees:** @cnnadozi
- **Description:** Remove unnecessary weights from the trained model. Compare new accuracy/size/speed vs baseline.
- **Notebook:** [M2_03_pruning.m](../notebooks/M2_03_pruning.m)
- **Write-up:** [M2_03_pruning.md](milestone_2/M2_03_pruning.md)

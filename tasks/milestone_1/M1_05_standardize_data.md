# Task 5: Standardize Data

## What I Got Done

- Calculated mean and standard deviation per signal position, using the training split only.
- Standardized the training data using those stats.
- Applied the same training mean and standard deviation to validation and test, no new stats calculated there.
- Saved standardized data to `data/standardized.mat` and the scaler to `data/scaler.mat`.
- Verified the saved scaler matches what standardizing by hand produces.

## Results

- Training data's mean is ~0 and standard deviation is ~1 after standardization.
- Validation and test are scaled using training stats, not their own, so their mean/std won't be exactly 0/1. Expected.
- `scaler.mat` holds `mu` and `sigma` for reuse in later scripts.

## Why This Matters

- Standardizing puts the raw values (originally -45 to 39) on a consistent scale, helping the model train better.
- Using only training stats for all splits avoids data leakage.
- Saving the scaler keeps preprocessing identical across every model trained later, so comparisons stay fair.

## Final Note

Data is standardized and the scaler is saved. Next step is training the baseline model.
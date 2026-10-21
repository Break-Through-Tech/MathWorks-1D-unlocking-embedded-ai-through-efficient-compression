# Task 1: Download Data

**Owner:** Huy  
**Milestone:** September  
**Issue:** [#7](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/issues/7)  
**PR:** [#9](https://github.com/Break-Through-Tech/MathWorks-1D-unlocking-embedded-ai-through-efficient-compression/pull/9)  
**Notebook:** [M1_01_load_data.ipynb](../notebooks/M1_01_load_data.ipynb)

## What I Got Done

- The `.mat` files were already committed to `data/`, so I focused on getting them opening correctly in our notebook/workspace.
- Wrote a notebook that loads `train.mat`, `val.mat`, and `test.mat` with `scipy.io.loadmat`.
- Printed a summary table for each split: number of signals, signal length, dtype, per-class counts, and min/max/mean/std.
- Added checks that the data matches the schema in [`data/README.md`](../data/README.md).
- Filled in `requirements.txt` (it was empty) with numpy, scipy, pandas, matplotlib, and jupyter.

## Results

| Split | Shape | Normal | InnerRaceFault | OuterRaceFault |
|-------|-------|-------:|---------------:|---------------:|
| Train | 393 x 5,000 | 131 | 131 | 131 |
| Validation | 27 x 5,000 | 9 | 9 | 9 |
| Test | 102 x 5,000 | 34 | 34 | 34 |

- 522 signals in total, all `float64`.
- No NaN or Inf values.
- Every split has all three classes and they are balanced.

## How to Run

1. `pip install -r requirements.txt`
2. Run `notebooks/M1_01_load_data.ipynb` top to bottom. It works from either `notebooks/` or the repo root and should print "All splits loaded and verified."

## Final Note

No plots or standardization here on purpose, since those were Task 4 (visualize signals) and Task 5 (standardize data).

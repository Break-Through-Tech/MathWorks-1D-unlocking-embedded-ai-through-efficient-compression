# Task 3: Equal Row Counts

## What I Got Done

- Loaded the train, validation, and test sets from the `.mat` files.
- Counted how many samples there are in each of the three classes: Normal, InnerRaceFault, and OuterRaceFault.
- Compared the counts across classes to check whether they are equal.
- Repeated the count for each split (train, validation, and test).

## Results

| Split | Normal | InnerRaceFault | OuterRaceFault | Total |
|---|---|---|---|---|
| Train | 131 | 131 | 131 | 393 |
| Validation | 9 | 9 | 9 | 27 |
| Test | 34 | 34 | 34 | 102 |

- All three classes have exactly the same number of samples in every split.
- The splits are about 75% train, 5% validation, and 20% test, which matches the dataset README.
- There is no class imbalance, so no resampling or class weighting is needed.

## Why This Matters

- Because the classes are balanced, accuracy is a fair metric for this project.
- A model that guesses randomly would score about 33% (1 out of 3), so that is the baseline to beat.
- The validation set is small (27 samples), so each validation sample is worth about 3.7% accuracy. Validation results may be jumpy.

## Final Note

The dataset is balanced and ready for the next steps. The next step is to visualize example signals from each class, then standardize the data for model training.
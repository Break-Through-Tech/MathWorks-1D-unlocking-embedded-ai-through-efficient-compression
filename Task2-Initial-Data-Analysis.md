# Task 2: Initial Data Analysis

## What I Got Done

- Loaded the train, validation, and test sets from the `.mat` files.
- Checked the data shape and label format.
- Verified that the signals are 5,000 samples long.
- Confirmed the three classes are present: Normal, InnerRaceFault, and OuterRaceFault.
- Checked for missing or invalid values.
- Looked at signal-level statistics to identify unusual patterns.

## Results

- No missing values were found in the dataset.
- No invalid values were found in the dataset.
- The data is balanced across the three classes.
- The splits match the expected structure and sizes.

## Outlier Check

- I reviewed signal statistics such as mean, standard deviation, and maximum absolute value.
- I flagged a few signals that were unusually high or low compared with the rest.
- These signals were not removed automatically because they may represent real fault behavior.
- Only corrupted samples would be removed if they had broken values.

## Final Note

The dataset is clean enough to move on to standardization and model training. The next step is to standardize the training data and apply the same transformation to the validation and test sets!
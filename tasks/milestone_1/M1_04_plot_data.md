# Task 4: Plot Example Signals per Category

## What I Got Done

- Loaded the train split and pulled example signals for Normal, InnerRaceFault, and OuterRaceFault.
- Plotted one example signal per class in the time domain.
- Plotted three overlaid examples per class to check if patterns hold across multiple samples.
- Ran an FFT on the same signals to check for differences in the frequency domain.

## Results

- Normal: amplitude stays small, -3 to 3, no sharp spikes.
- InnerRaceFault: amplitude spikes up to ~25, clear periodic pattern.
- OuterRaceFault: similar range to Normal, hard to tell apart by eye.
- In the FFT, OuterRaceFault has a distinct peak around 2,500-3,000 Hz that Normal does not have.
- Normal and InnerRaceFault patterns hold up consistently across examples. OuterRaceFault is less consistent example to example.

## Why This Matters

- InnerRaceFault is easy to classify from the raw signal alone.
- OuterRaceFault and Normal look alike in the raw signal, so a model relying only on raw amplitude may struggle here.
- The FFT shows OuterRaceFault is separable in frequency, even though it isn't in time.
- If the model misclassifies any class, OuterRaceFault vs. Normal is the most likely spot to check.

## Final Note

InnerRaceFault is easy to separate visually. OuterRaceFault is not, in time, but is in frequency. Next step is standardizing the data for model training.
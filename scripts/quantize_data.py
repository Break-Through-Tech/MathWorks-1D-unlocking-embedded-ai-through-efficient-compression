"""Quantize the standardized bearing signals to symmetric signed 8-bit values.

Run from any directory with ``python scripts/quantize_data.py``. The scale is
calibrated on the training split only and reused for validation and test.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
QMAX = 127


def main() -> None:
    standardized = loadmat(DATA_DIR / "standardized.mat")
    raw_train = loadmat(DATA_DIR / "train.mat")["trainData"]
    validation = loadmat(DATA_DIR / "val.mat")

    train = standardized["trainData"]
    test = standardized["testData"]
    if train.shape != raw_train.shape or train.shape[1] != test.shape[1]:
        raise ValueError("Train and test signal dimensions do not match")

    # Match the per-column transformation used to create standardized.mat.
    mean = raw_train.mean(axis=0)
    std = raw_train.std(axis=0)
    if np.any(std == 0):
        raise ValueError("Cannot standardize a constant training column")
    if not np.allclose((raw_train - mean) / std, train, rtol=1e-12, atol=1e-12):
        raise ValueError("standardized.mat does not match the training statistics")
    val = (validation["valData"] - mean) / std

    for split in (train, val, test):
        if not np.isfinite(split).all():
            raise ValueError("Signals contain non-finite values")

    scale = float(np.max(np.abs(train)) / QMAX)
    if scale == 0:
        raise ValueError("Training signals have no nonzero values")

    def quantize(values: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(values / scale), -QMAX, QMAX).astype(np.int8)

    output = {
        "trainData": quantize(train),
        "trainLabels": standardized["trainLabels"],
        "valData": quantize(val),
        "valLabels": validation["valLabels"],
        "testData": quantize(test),
        "testLabels": standardized["testLabels"],
        "scale": np.array([[scale]], dtype=np.float64),
        "zeroPoint": np.array([[0]], dtype=np.int8),
        "trainMean": mean.reshape(1, -1),
        "trainStd": std.reshape(1, -1),
    }
    destination = DATA_DIR / "quantized.mat"
    savemat(destination, output)
    print(f"Saved {destination}")
    for name, values in (("train", train), ("val", val), ("test", test)):
        clipped = np.count_nonzero(np.abs(values) > QMAX * scale)
        reconstructed = output[f"{name}Data"].astype(np.float64) * scale
        mae = np.mean(np.abs(values - reconstructed))
        print(f"{name}: clipped {clipped}/{values.size}, mean absolute error {mae:.6f}")


if __name__ == "__main__":
    main()


import h5py
import numpy as np

file = "YOUR FILE PATH HERE"
import os

with h5py.File(file, "r") as f:

    ds = f["US"]["US_DATASET0000"]

    real = ds["data"]["real"][:]
    imag = ds["data"]["imag"][:]

    print("Real shape:", real.shape)
    print("Imag shape:", imag.shape)

    print("Real min/max:", real.min(), real.max())
    print("Imag min/max:", imag.min(), imag.max())

    print("Imaginary absolute maximum:",
          np.max(np.abs(imag)))

    print("Sampling frequency:",
          ds["sampling_frequency"][0])

    print("Modulation frequency:",
          ds["modulation_frequency"][0])

    print("Sound speed:",
          ds["sound_speed"][0])

    print("Initial time:",
          ds["initial_time"][0])

    print("Angles:", ds["angles"][:])

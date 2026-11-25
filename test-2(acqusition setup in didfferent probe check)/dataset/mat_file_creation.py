import h5py
import numpy as np
from scipy.io import savemat

input_file = "YOUR INPUT FILE PATH HERE"
output_file = "YOUR OUTPUT FILE PATH HERE"

with h5py.File(input_file, "r") as f:

    ds = f["US"]["US_DATASET0000"]

    # Extract real RF data
    rf_raw = ds["data"]["real"][:]

    # Convert from (angles, elements, samples)
    # to (samples, elements, angles)
    rf = np.transpose(rf_raw, (2, 1, 0))

    # Acquisition metadata
    angles = ds["angles"][:]
    fs = float(ds["sampling_frequency"][0])
    c = float(ds["sound_speed"][0])
    initial_time = float(ds["initial_time"][0])
    probe_geometry = ds["probe_geometry"][:]
    prf = float(ds["PRF"][0])

savemat(output_file, {
    "rf": rf,
    "angles": angles,
    "fs": fs,
    "c": c,
    "initial_time": initial_time,
    "probe_geometry": probe_geometry,
    "prf": prf
}, do_compression=True)

print("MAT file created:", output_file)
print("RF dimensions:", rf.shape)
print("Sampling frequency:", fs)
print("Number of angles:", len(angles))

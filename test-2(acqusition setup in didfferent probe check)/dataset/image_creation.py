
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import os

# Input HDF5 file
input_file = "YOUR  INPUT FILE PATH HERE"
# Output directory
output_dir = r"PICMUS_DAS_Images"
os.makedirs(output_dir, exist_ok=True)

with h5py.File(input_file, "r") as f:

    ds = f["US"]["US_DATASET0000"]

    real = ds["data"]["real"][:]
    imag = ds["data"]["imag"][:]

    x_axis = ds["scan"]["x_axis"][:]
    z_axis = ds["scan"]["z_axis"][:]

    plane_waves = ds["number_plane_waves"][:]

    print("Data shape:", real.shape)
    print("Number of configurations:", real.shape[0])
    print("Imaginary maximum:", np.max(np.abs(imag)))

    for i in range(real.shape[0]):

        rf = real[i]

        # Envelope detection
        if np.max(np.abs(imag[i])) > 1e-10:
            envelope = np.abs(real[i] + 1j * imag[i])
        else:
            envelope = np.abs(hilbert(rf, axis=-1))

        # Normalize
        envelope = envelope / (np.max(envelope) + 1e-12)

        # Log compression
        image_db = 20 * np.log10(envelope + 1e-12)

        # Display range
        image_db = np.clip(image_db, -60, 0)

        # Save image
        plt.figure(figsize=(7, 8))

        plt.imshow(
            image_db.T,
            extent=[
                x_axis[0] * 1000,
                x_axis[-1] * 1000,
                z_axis[-1] * 1000,
                z_axis[0] * 1000
            ],
            cmap="gray",
            vmin=-60,
            vmax=0,
            aspect="auto"
        )

        plt.xlabel("Lateral distance (mm)")
        plt.ylabel("Depth (mm)")
        plt.title(f"DAS-RF: {plane_waves[i]:g} Plane Waves")

        plt.colorbar(label="Amplitude (dB)")
        plt.tight_layout()

        output_file = os.path.join(
            output_dir,
            f"PICMUS_DAS_{i+1}.png"
        )

        plt.savefig(output_file, dpi=300)
        plt.close()

        print("Saved:", output_file)

print("Conversion completed!")


import numpy as np
import cupy as cp


def preprocessing(sig, USHEADER, param, n_ang_ss=0):

    # ========================================================
    # DEFAULT SETTINGS
    # ========================================================

    if n_ang_ss is None:
        n_ang_ss = 0

    lens_delay = 96

    # ========================================================
    # ENSURE DATA HAS THREE DIMENSIONS
    # ========================================================

    sig = np.asarray(sig)
    sig = np.squeeze(sig)

    if sig.ndim == 2:
        sig = np.reshape(
            sig,
            (sig.shape[0], sig.shape[1], 1)
        )

    if sig.ndim != 3:
        raise ValueError(
            "Signal must have three dimensions: "
            "[time x channels x angles]."
        )

    # ========================================================
    # TRANSFER SIGNAL TO NVIDIA GPU
    # ========================================================

    sig = cp.asarray(sig, dtype=cp.float64)

    # ========================================================
    # LOAD ACQUISITION PARAMETERS
    # Angle metadata is small, so keep it on CPU.
    # ========================================================

    tx_angles = np.asarray(
        USHEADER['xmitAngles'],
        dtype=np.float64
    ).reshape(-1)

    n_ele = sig.shape[1]
    n_angles = sig.shape[2]

    fs = float(param['fs'])
    c = float(param['c'])
    pitch = float(param['pitch'])

    # ========================================================
    # CHECK DIMENSIONS
    # ========================================================

    if len(tx_angles) != n_angles:
        raise ValueError(
            "Number of transmission angles does not match "
            "the third dimension of ultrasound data."
        )

    # ========================================================
    # ANGLE SUBSAMPLING
    # ========================================================

    if n_ang_ss == 1:

        ang_ind = np.array([
            np.argmin(np.abs(tx_angles))
        ])

    elif n_ang_ss > 1 and n_ang_ss < n_angles:

        ang_ss = np.linspace(
            tx_angles[0],
            tx_angles[-1],
            int(n_ang_ss)
        )

        ang_ind = np.zeros(
            int(n_ang_ss),
            dtype=int
        )

        for k in range(int(n_ang_ss)):

            ang_ind[k] = np.argmin(
                np.abs(tx_angles - ang_ss[k])
            )

    elif n_ang_ss == 0 or n_ang_ss == n_angles:

        ang_ind = np.arange(n_angles)

    else:

        raise ValueError(
            "Invalid number of subsampled angles."
        )

    # ========================================================
    # RETAIN SELECTED ANGLES ON GPU
    # ========================================================

    ang_ind_gpu = cp.asarray(ang_ind)

    sig = sig[:, :, ang_ind_gpu]

    tx_angles = tx_angles[ang_ind]

    # ========================================================
    # 1. LENS DELAY CORRECTION
    # ========================================================

    if sig.shape[0] <= lens_delay:

        raise ValueError(
            "Insufficient signal samples for lens delay correction."
        )

    sig = sig[lens_delay:, :, :]

    # ========================================================
    # 2. ANGLE-DEPENDENT DELAY CORRECTION
    # ========================================================

    n_t = sig.shape[0]

    dt = 1.0 / fs

    # Calculate delay indices on GPU

    tx_angles_gpu = cp.asarray(
        tx_angles,
        dtype=cp.float64
    )

    delay = cp.abs(
        ((n_ele - 1) / 2)
        * pitch
        * cp.sin(cp.deg2rad(tx_angles_gpu))
        / c
    )

    delay_ind = cp.floor(
        delay / dt
    ).astype(cp.int64)

    # Apply delay correction

    for k in range(len(tx_angles)):

        d = int(delay_ind[k].item())

        if d == 0:
            continue

        if d >= n_t:

            sig[:, :, k] = 0

            continue

        # Shift signals to correct transmission delay

        sig[0:n_t-d, :, k] = (
            sig[d:n_t, :, k]
        )

        # Zero-pad the end

        sig[n_t-d:n_t, :, k] = 0

    # ========================================================
    # 3. CHANNEL 126 CORRECTION
    # ========================================================

    if sig.shape[1] >= 126:

        sig[:, 125, :] = 2.0 * sig[:, 125, :]

    else:

        print(
            "WARNING: Channel 126 does not exist. "
            "Skipping channel correction."
        )

    # ========================================================
    # 4. UPDATE FK MIGRATION PARAMETERS
    # ========================================================

    param['TXangle'] = cp.deg2rad(tx_angles_gpu)

    # ========================================================
    # SYNCHRONIZE GPU
    # ========================================================

    cp.cuda.Stream.null.synchronize()

    # ========================================================
    # OUTPUT INFORMATION
    # ========================================================

    print("\nCUDA Ultrasound preprocessing completed.")

    print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"])

    print("Remaining samples:", sig.shape[0])

    print("Number of channels:", sig.shape[1])

    print("Number of transmission angles:", sig.shape[2])

    print("Output device: CUDA GPU")

    print("Output dtype:", sig.dtype)

    print("Output shape:", sig.shape)

    return sig, param

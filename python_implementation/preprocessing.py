
import numpy as np


def preprocessing(sig, USHEADER, param, n_ang_ss=0):

    # ========================================================
    # DEFAULT SETTINGS
    # ========================================================

    if n_ang_ss is None:
        n_ang_ss = 0

    lens_delay = 96

    # ========================================================
    # ENSURE DATA HAS THREE DIMENSIONS
    # MATLAB: sig = squeeze(sig)
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

    sig = sig.astype(np.float64)

    # ========================================================
    # LOAD ACQUISITION PARAMETERS
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

        # Select angle closest to zero degrees

        ang_ind = np.array([
            np.argmin(np.abs(tx_angles))
        ])

    elif n_ang_ss > 1 and n_ang_ss < n_angles:

        # Select evenly distributed transmission angles

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

        # Retain all transmission angles

        ang_ind = np.arange(n_angles)

    else:

        raise ValueError(
            "Invalid number of subsampled angles."
        )

    # ========================================================
    # RETAIN SELECTED ANGLES
    # ========================================================

    sig = sig[:, :, ang_ind]

    tx_angles = tx_angles[ang_ind]

    # ========================================================
    # 1. LENS DELAY CORRECTION
    # MATLAB: sig = sig(lens_delay+1:end,:,:)
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

    delay = np.abs(
        ((n_ele - 1) / 2)
        * pitch
        * np.sin(np.deg2rad(tx_angles))
        / c
    )

    delay_ind = np.floor(
        delay / dt
    ).astype(int)

    for k in range(len(tx_angles)):

        d = delay_ind[k]

        if d == 0:
            continue

        if d >= n_t:

            sig[:, :, k] = 0

            continue

        # Shift signals to correct transmission delay
        # MATLAB:
        # sig(1:n_t-d,:,k) = sig(d+1:n_t,:,k)

        sig[0:n_t-d, :, k] = (
            sig[d:n_t, :, k]
        )

        # Zero-pad the end
        # MATLAB:
        # sig(n_t-d+1:n_t,:,k) = 0

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

    param['TXangle'] = np.deg2rad(tx_angles)

    # ========================================================
    # OUTPUT INFORMATION
    # ========================================================

    print("\nUltrasound preprocessing completed.")

    print(
        "Remaining samples:",
        sig.shape[0]
    )

    print(
        "Number of channels:",
        sig.shape[1]
    )

    print(
        "Number of transmission angles:",
        sig.shape[2]
    )

    return sig, param

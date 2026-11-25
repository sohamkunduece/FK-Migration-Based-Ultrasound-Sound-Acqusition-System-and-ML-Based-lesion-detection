
import numpy as np
import cupy as cp

from scipy.signal import butter, freqz


# ============================================================
# MATLAB ROUND
# ============================================================

def matlab_round(x):

    if isinstance(x, cp.ndarray):
        return cp.sign(x) * cp.floor(cp.abs(x) + 0.5)

    x = np.asarray(x)

    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ============================================================
# FK MIGRATION - CUDA VERSION
# ============================================================

def fkmig(SIG, param):

    # --------------------------------------------------------
    # 1. TRANSFER INPUT TO GPU
    # --------------------------------------------------------

    SIG = cp.asarray(SIG)

    if SIG.ndim == 2:
        SIG = SIG[:, :, cp.newaxis]

    if SIG.ndim != 3:
        raise ValueError("SIG must be a 2-D or 3-D array.")

    nt, nx, Nframes = SIG.shape

    # --------------------------------------------------------
    # 2. INPUT PARAMETERS
    # --------------------------------------------------------

    if 'c' not in param:
        param['c'] = 1540.0

    c = float(param['c'])

    if 'fs' not in param:
        raise ValueError(
            "A sample frequency (fs) must be specified."
        )

    fs = float(param['fs'])

    if 't0' not in param:
        param['t0'] = np.zeros(Nframes)

    t0 = np.asarray(
        cp.asnumpy(param['t0'])
        if isinstance(param['t0'], cp.ndarray)
        else param['t0'],
        dtype=float
    ).flatten()

    if t0.size == 1:
        t0 = t0 * np.ones(Nframes)

    if t0.size != Nframes:
        raise ValueError(
            "PARAM.t0 must be scalar or have length Nframes."
        )

    if 'pitch' not in param:
        raise ValueError("A pitch value is required.")

    pitch = float(param['pitch'])

    if 'TXangle' not in param:
        param['TXangle'] = 0.0

    TXangle = np.asarray(
        cp.asnumpy(param['TXangle'])
        if isinstance(param['TXangle'], cp.ndarray)
        else param['TXangle'],
        dtype=float
    ).flatten()

    if TXangle.size == 1:
        TXangle = TXangle * np.ones(Nframes)

    if TXangle.size != Nframes:
        raise ValueError(
            "PARAM.TXangle must be scalar or have length Nframes."
        )

    if 'p' not in param:
        param['p'] = 0

    p = int(param['p'])

    if p < 0 or p != param['p']:
        raise ValueError(
            "PARAM.p must be a nonnegative integer."
        )

    if 'passband' in param:
        wn = np.asarray(
            param['passband'],
            dtype=float
        ).flatten()
    else:
        wn = np.array([0.0, 1.0])

    if (
        wn.size != 2
        or wn[0] >= wn[1]
        or wn[0] < 0
        or wn[1] > 1
    ):
        raise ValueError("Invalid PARAM.passband.")

    if 'ncol' not in param:
        param['ncol'] = nx

    ncol = int(param['ncol'])

    # --------------------------------------------------------
    # 3. CONVERT TO DOUBLE PRECISION
    # --------------------------------------------------------

    SIG = SIG.astype(cp.float64)

    # --------------------------------------------------------
    # 4. TEMPORAL SHIFT
    # --------------------------------------------------------

    ntshift = int(
        np.max(matlab_round(t0 * fs))
    )

    # --------------------------------------------------------
    # 5. ZERO PADDING
    # --------------------------------------------------------

    if p:
        ntFFT = int(matlab_round(1.5 * nt)) + ntshift
    else:
        ntFFT = 4 * nt + ntshift

    if ntFFT % 2 == 1:
        ntFFT += 1

    factor = 1.5

    nxFFT = int(np.ceil(factor * nx))

    if nxFFT % 2 == 1:
        nxFFT += 1

    dx = int(
        matlab_round(factor * (ncol - nx))
    )

    # --------------------------------------------------------
    # 6. FILTER DESIGN
    # CPU DESIGN, GPU APPLICATION
    # --------------------------------------------------------

    if not np.array_equal(wn, np.array([0.0, 1.0])):

        b, a = butter(
            5,
            wn,
            btype='bandpass'
        )

        w = np.linspace(
            0.0,
            np.pi,
            ntFFT // 2 + 1
        )

        _, h = freqz(
            b,
            a,
            worN=w
        )

        hf = np.abs(h)
        hf = hf / np.max(hf)

        hf = cp.asarray(
            hf,
            dtype=cp.float64
        )

    else:
        hf = 1.0

    # --------------------------------------------------------
    # 7. FREQUENCY AND SPATIAL COORDINATES
    # --------------------------------------------------------

    f0 = (
        cp.arange(
            ntFFT // 2 + 1,
            dtype=cp.float64
        )
        * fs / ntFFT
    )

    kx_vector = cp.concatenate((

        cp.arange(
            nxFFT // 2 + 1,
            dtype=cp.float64
        ),

        cp.arange(
            -nxFFT // 2 + 1,
            0,
            dtype=cp.float64
        )

    ))

    kx_vector = (
        kx_vector / pitch / nxFFT
    )

    # MATLAB meshgrid
    kx, f = cp.meshgrid(
        kx_vector,
        f0
    )

    # --------------------------------------------------------
    # 8. INITIALIZE MIGRATED SIGNAL
    # --------------------------------------------------------

    migSIG = cp.zeros(
        (ntFFT, nxFFT),
        dtype=cp.complex128
    )

    # --------------------------------------------------------
    # 9. TEMPORAL FFT
    # --------------------------------------------------------

    SIG = cp.fft.fft(
        SIG,
        n=ntFFT,
        axis=0
    )

    SIG = SIG[
        :ntFFT // 2 + 1,
        :, :
    ]

    # ========================================================
    # 10. PROCESS EACH FRAME
    # ========================================================

    for k in range(Nframes):

        SIGk = SIG[:, :, k].copy()

        sinA = np.sin(TXangle[k])
        cosA = np.cos(TXangle[k])

        # ----------------------------------------------------
        # ERM VELOCITY
        # ----------------------------------------------------

        v = c / np.sqrt(
            1 + cosA + sinA**2
        )

        # ----------------------------------------------------
        # STEERING ANGLE / DEPTH START COMPENSATION
        # ----------------------------------------------------

        if sinA != 0 or t0[k] != 0:

            dt = (
                sinA
                * (
                    (nx - 1) * (TXangle[k] < 0)
                    - cp.arange(
                        nx,
                        dtype=cp.float64
                    )
                )
                * pitch / c
            )

            tmp = cp.outer(
                f0,
                dt + t0[k]
            )

            SIGk = SIGk * cp.exp(
                -2j * cp.pi * tmp
            )

        # ----------------------------------------------------
        # SPATIAL FFT
        # ----------------------------------------------------

        SIGk = cp.fft.fft(
            SIGk,
            n=nxFFT,
            axis=1
        )

        # ----------------------------------------------------
        # STOLT FREQUENCY MAPPING
        # ----------------------------------------------------

        C = (
            (1 + cosA + sinA**2)
            / (1 + cosA)**1.5
        )

        fkz = v * cp.sqrt(
            kx**2
            + 4 * f**2 / c**2 * C**2
        )

        # ----------------------------------------------------
        # REMOVE EVANESCENT PARTS
        # ----------------------------------------------------

        isevanescent = (
            cp.abs(f) < c * cp.abs(kx)
        )

        SIGk[isevanescent] = 0

        # ----------------------------------------------------
        # STOLT INTERPOLATION
        # ----------------------------------------------------

        if p:

            SIGk = interpSINC(
                fs / ntFFT,
                SIGk,
                fkz[:, :nxFFT // 2 + 1],
                p,
                (np.array([0.0, 1.0]) + wn) / 2
            )

        else:

            SIGk = interpLIN(
                fs / ntFFT,
                SIGk,
                fkz,
                (np.array([0.0, 1.0]) + wn) / 2
            )

        # ----------------------------------------------------
        # OBLIQUITY FACTOR
        # ----------------------------------------------------

        safe_fkz = cp.where(
            fkz != 0,
            fkz,
            1.0
        )

        obliquity = cp.where(
            fkz != 0,
            f / safe_fkz,
            0.0
        )

        SIGk = SIGk * obliquity

        SIGk[0, 0] = 0

        # ----------------------------------------------------
        # FILTERING
        # ----------------------------------------------------

        if isinstance(hf, cp.ndarray):

            SIGk = SIGk * hf[:, cp.newaxis]

        elif hf != 1:

            SIGk = SIGk * hf

        # ----------------------------------------------------
        # HERMITIAN RECONSTRUCTION
        # ----------------------------------------------------

        lower = cp.conj(
            cp.concatenate((

                SIGk[
                    ntFFT // 2 - 1:0:-1,
                    0:1
                ],

                SIGk[
                    ntFFT // 2 - 1:0:-1,
                    -1:0:-1
                ]

            ), axis=1)
        )

        SIGk = cp.concatenate(
            (SIGk, lower),
            axis=0
        )

        # ----------------------------------------------------
        # FREQUENCY IFFT
        # ----------------------------------------------------

        SIGk = cp.fft.ifft(
            SIGk,
            axis=0
        )

        # ----------------------------------------------------
        # STEERING ANGLE COMPENSATION
        # ----------------------------------------------------

        if sinA != 0:

            C = sinA / (2 - cosA)

            dx_steer = (
                -C
                * cp.arange(
                    ntFFT,
                    dtype=cp.float64
                )
                / fs
                * c / 2
            )

            tmp = cp.outer(
                dx_steer,
                kx[0, :]
            )

            SIGk = SIGk * cp.exp(
                -2j * cp.pi * tmp
            )

        # ----------------------------------------------------
        # COMPOUNDING
        # ----------------------------------------------------

        migSIG = (
            (k * migSIG + SIGk) / (k + 1)
        )

    # ========================================================
    # 11. INTERPOLATION IN KX DIRECTION
    # ========================================================

    if ncol > nx:

        nyqst = nxFFT // 2 + 1

        migSIG = cp.concatenate((

            migSIG[:, :nyqst],

            cp.zeros(
                (ntFFT, dx),
                dtype=migSIG.dtype
            ),

            migSIG[:, nyqst:]

        ), axis=1)

        migSIG[:, nyqst - 1] /= 2

        migSIG[:, nyqst + dx - 1] = (
            migSIG[:, nyqst - 1]
        )

        migSIG = migSIG * ncol / nx

        nx = ncol

    # ========================================================
    # 12. FINAL SPATIAL IFFT
    # ========================================================

    migSIG = cp.fft.ifft(
        migSIG,
        axis=1
    ).real

    migSIG = migSIG[
        ntshift:ntshift + nt,
        :nx
    ]

    # ========================================================
    # 13. GRID COORDINATES
    # ========================================================

    param['x'] = (
        cp.arange(
            nx,
            dtype=cp.float64
        ) - (nx - 1) / 2
    ) * pitch

    param['z'] = (
        cp.arange(
            nt,
            dtype=cp.float64
        ) + ntshift
    ) * c / 2 / fs

    # --------------------------------------------------------
    # GPU SYNCHRONIZATION
    # --------------------------------------------------------

    cp.cuda.Stream.null.synchronize()

    print("\nCUDA FK migration completed.")

    print(
        "GPU:",
        cp.cuda.runtime.getDeviceProperties(0)["name"]
    )

    print("Output shape:", migSIG.shape)
    print("Output dtype:", migSIG.dtype)
    print("Output device: CUDA GPU")

    return migSIG, param


# ============================================================
# INTERPOLATION 1: CUDA LINEAR INTERPOLATION
# ============================================================

def interpLIN(dx, y, xi, wn):

    siz = y.shape

    yi = cp.zeros(
        siz,
        dtype=cp.result_type(
            y.dtype,
            cp.float64
        )
    )

    n1 = int(matlab_round(
        wn[0] * (siz[0] - 1) + 1
    ))

    n2 = int(matlab_round(
        wn[1] * (siz[0] - 1) + 1
    ))

    idx = xi / dx + 1

    I = idx > (siz[0] - 1)

    idx = idx.copy()
    idx[I] = 1

    idxf = cp.floor(idx).astype(cp.int64)

    start = n1 - 1
    stop = n2

    for k in range(siz[1]):

        idxfk = idxf[start:stop, k]

        idxk = (
            idxfk - idx[start:stop, k]
        )

        yi[start:stop, k] = (
            y[idxfk - 1, k] * (idxk + 1)
            - y[idxfk, k] * idxk
        )

    yi[I] = 0

    return yi


# ============================================================
# INTERPOLATION 2: CUDA P-POINT SINC INTERPOLATION
# ============================================================

def interpSINC(dx, y, xi, p, wn):

    def sinc0(x):

        return 1 + (8 / cp.pi - 4) * x**2

    siz = y.shape

    yi = cp.zeros(
        siz,
        dtype=cp.complex128
    )

    n1 = int(matlab_round(
        wn[0] * (siz[0] - 1) + 1
    ))

    n2 = int(matlab_round(
        wn[1] * (siz[0] - 1) + 1
    ))

    xi = xi / dx + 1

    idxr = matlab_round(xi).astype(cp.int64)

    delta_n = xi - idxr

    SINmat = cp.sin(cp.pi * delta_n)

    EXPmat = (
        cp.cos(cp.pi * delta_n)
        - 1j * SINmat
    )

    mp = (
        list(range(
            int(np.floor(-p / 2 + 1)),
            0
        ))
        +
        list(range(
            1,
            int(np.floor(p / 2)) + 1
        ))
    )

    start = n1 - 1
    stop = n2

    # --------------------------------------------------------
    # PROCESS FREQUENCY COLUMNS
    # --------------------------------------------------------

    for k in range(1, siz[1] // 2 + 1):

        yk1 = y[:, k]

        yk2 = y[:, siz[1] - k]

        idxrk = idxr[start:stop, k].copy()

        delta_nk = delta_n[start:stop, k].copy()

        for m in mp:

            C = 1.0 / (delta_nk - m)

            idxk = idxrk + m

            I = (
                (idxk < 1)
                |
                (idxk > siz[0])
            )

            C[I] = 0

            idxk[I] = 1

            yi[start:stop, k] += (
                C * yk1[idxk - 1]
            )

            if k < siz[1] // 2:

                yi[
                    start:stop,
                    siz[1] - k
                ] += (
                    C * yk2[idxk - 1]
                )

        yi[start:stop, k] = (
            1 / cp.pi
            * SINmat[start:stop, k]
            * yi[start:stop, k]
        )

        if k < siz[1] // 2:

            yi[
                start:stop,
                siz[1] - k
            ] = (
                1 / cp.pi
                * SINmat[start:stop, k]
                * yi[start:stop, siz[1] - k]
            )

        # ----------------------------------------------------
        # IDEM FOR m = 0
        # ----------------------------------------------------

        idxrk = idxrk.copy()

        I = idxrk > siz[0]

        idxrk[I] = 1

        delta_nk[I] = (
            (4 - 8 / cp.pi) ** -0.5
        )

        S = sinc0(delta_nk)

        yi[start:stop, k] += (
            yk1[idxrk - 1] * S
        )

        if k < siz[1] // 2:

            yi[
                start:stop,
                siz[1] - k
            ] += (
                yk2[idxrk - 1] * S
            )

        yi[start:stop, k] *= (
            EXPmat[start:stop, k]
        )

        if k < siz[1] // 2:

            yi[
                start:stop,
                siz[1] - k
            ] *= (
                EXPmat[start:stop, k]
            )

    return yi


import numpy as np
from scipy.signal import butter, freqz


# ============================================================
# MATLAB ROUND
# MATLAB rounds halfway values away from zero.
# NumPy round uses ties-to-even, so we implement MATLAB behavior.
# ============================================================

def matlab_round(x):
    x = np.asarray(x)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


# ============================================================
# MATLAB SIZE(SIG)
# ============================================================

def fkmig(SIG, param):

    SIG = np.asarray(SIG)

    if SIG.ndim == 2:
        SIG = SIG[:, :, np.newaxis]

    if SIG.ndim != 3:
        raise ValueError("SIG must be a 2-D or 3-D array.")

    nt, nx, Nframes = SIG.shape

    # --------------------------------------------------------
    # 1. INPUT PARAMETERS
    # --------------------------------------------------------

    if 'c' not in param:
        param['c'] = 1540.0

    c = param['c']

    if 'fs' not in param:
        raise ValueError(
            "A sample frequency (fs) must be specified."
        )

    if 't0' not in param:
        param['t0'] = np.zeros(Nframes)

    t0 = np.asarray(param['t0'], dtype=float).flatten()

    if t0.size == 1:
        t0 = t0 * np.ones(Nframes)

    if t0.size != Nframes:
        raise ValueError(
            "PARAM.t0 must be scalar or have length Nframes."
        )

    if 'pitch' not in param:
        raise ValueError("A pitch value is required.")

    if 'TXangle' not in param:
        param['TXangle'] = 0.0

    TXangle = np.asarray(
        param['TXangle'], dtype=float
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
        raise ValueError("PARAM.p must be a nonnegative integer.")

    if 'passband' in param:
        wn = np.asarray(
            param['passband'], dtype=float
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
    # 2. CONVERT INPUT TO DOUBLE
    # MATLAB: SIG = double(SIG);
    # --------------------------------------------------------

    SIG = SIG.astype(np.float64)

    # --------------------------------------------------------
    # 3. TEMPORAL SHIFT
    # --------------------------------------------------------

    ntshift = int(np.max(matlab_round(t0 * param['fs'])))

    # --------------------------------------------------------
    # 4. ZERO PADDING BEFORE FFT
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

    dx = int(matlab_round(factor * (ncol - nx)))

    # --------------------------------------------------------
    # 5. CREATE FILTER
    # --------------------------------------------------------

    if not np.array_equal(wn, np.array([0.0, 1.0])):

        b, a = butter(5, wn, btype='bandpass')

        # MATLAB freqz(b,a,ntFFT/2+1)
        # Evaluate over [0, pi], including both endpoints.
        w = np.linspace(
            0.0, np.pi, ntFFT // 2 + 1
        )

        _, h = freqz(b, a, worN=w)

        hf = np.abs(h)
        hf = hf / np.max(hf)

    else:
        hf = 1.0

    # --------------------------------------------------------
    # 6. FREQUENCY AND SPATIAL COORDINATES
    # --------------------------------------------------------

    f0 = (
        np.arange(ntFFT // 2 + 1, dtype=float)
        * param['fs'] / ntFFT
    )

    kx_vector = np.concatenate((
        np.arange(nxFFT // 2 + 1),
        np.arange(-nxFFT // 2 + 1, 0)
    ))

    kx_vector = (
        kx_vector / param['pitch'] / nxFFT
    )

    # MATLAB:
    # [kx,f] = meshgrid(kx,f0);

    kx, f = np.meshgrid(kx_vector, f0)

    # --------------------------------------------------------
    # 7. INITIALIZE MIGRATED SIGNAL
    # --------------------------------------------------------

    migSIG = np.zeros(
        (ntFFT, nxFFT), dtype=np.complex128
    )

    # --------------------------------------------------------
    # 8. TEMPORAL FFT
    # MATLAB: fft(SIG,ntFFT)
    # --------------------------------------------------------

    SIG = np.fft.fft(SIG, n=ntFFT, axis=0)

    # Keep positive temporal frequencies.
    SIG = SIG[:ntFFT // 2 + 1, :, :]

    # ========================================================
    # 9. PROCESS EACH FRAME
    # ========================================================

    for k in range(Nframes):

        # MATLAB: SIGk = SIG(:,:,k)
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
        # COMPENSATE FOR STEERING ANGLE / DEPTH START
        # ----------------------------------------------------

        if sinA != 0 or t0[k] != 0:

            dt = (
                sinA
                * (
                    (nx - 1) * (TXangle[k] < 0)
                    - np.arange(nx)
                )
                * param['pitch'] / c
            )

            tmp = np.outer(
                f0, dt + t0[k]
            )

            SIGk = SIGk * np.exp(
                -2j * np.pi * tmp
            )

        # ----------------------------------------------------
        # SPATIAL FFT
        # MATLAB: fft(SIGk,nxFFT,2)
        # ----------------------------------------------------

        SIGk = np.fft.fft(
            SIGk, n=nxFFT, axis=1
        )

        # ----------------------------------------------------
        # STOLT FREQUENCY MAPPING
        # ----------------------------------------------------

        C = (
            (1 + cosA + sinA**2)
            / (1 + cosA)**1.5
        )

        fkz = v * np.sqrt(
            kx**2 + 4 * f**2 / c**2 * C**2
        )

        # ----------------------------------------------------
        # REMOVE EVANESCENT PARTS
        # ----------------------------------------------------

        with np.errstate(
            divide='ignore',
            invalid='ignore'
        ):

            isevanescent = (
                np.abs(f) / np.abs(kx) < c
            )

        SIGk[isevanescent] = 0

        # ----------------------------------------------------
        # INTERPOLATION: f -> fkz
        # ----------------------------------------------------

        if p:

            SIGk = interpSINC(
                param['fs'] / ntFFT,
                SIGk,
                fkz[:, :nxFFT // 2 + 1],
                p,
                (np.array([0.0, 1.0]) + wn) / 2
            )

        else:

            SIGk = interpLIN(
                param['fs'] / ntFFT,
                SIGk,
                fkz,
                (np.array([0.0, 1.0]) + wn) / 2
            )

        # ----------------------------------------------------
        # OBLIQUITY FACTOR
        # ----------------------------------------------------

        with np.errstate(
            divide='ignore',
            invalid='ignore'
        ):

            SIGk = SIGk * f / fkz

        SIGk[0, 0] = 0

        # ----------------------------------------------------
        # FILTERING
        # ----------------------------------------------------

        if np.ndim(hf) > 0:

            SIGk = SIGk * hf[:, np.newaxis]

        elif hf != 1:

            SIGk = SIGk * hf

        # ----------------------------------------------------
        # FREQUENCY IFFT
        # HERMITIAN RECONSTRUCTION
        # ----------------------------------------------------

        lower = np.conj(
            np.concatenate((
                SIGk[ntFFT // 2 - 1:0:-1, 0:1],
                SIGk[
                    ntFFT // 2 - 1:0:-1,
                    -1:0:-1
                ]
            ), axis=1)
        )

        SIGk = np.concatenate(
            (SIGk, lower),
            axis=0
        )

        SIGk = np.fft.ifft(SIGk, axis=0)

        # ----------------------------------------------------
        # COMPENSATE FOR STEERING ANGLE
        # ----------------------------------------------------

        if sinA != 0:

            C = sinA / (2 - cosA)

            dx_steer = (
                -C
                * np.arange(ntFFT)
                / param['fs']
                * c / 2
            )

            tmp = np.outer(
                dx_steer, kx[0, :]
            )

            SIGk = SIGk * np.exp(
                -2j * np.pi * tmp
            )

        # ----------------------------------------------------
        # COMPOUNDING
        # MATLAB: ((k-1)*migSIG + SIGk)/k
        # ----------------------------------------------------

        migSIG = (
            (k * migSIG + SIGk) / (k + 1)
        )

    # ========================================================
    # 10. INTERPOLATION IN KX DIRECTION
    # ========================================================

    if ncol > nx:

        nyqst = nxFFT // 2 + 1

        migSIG = np.concatenate((
            migSIG[:, :nyqst],
            np.zeros(
                (ntFFT, dx),
                dtype=migSIG.dtype
            ),
            migSIG[:, nyqst:]
        ), axis=1)

        # MATLAB: migSIG(:,nyqst) = migSIG(:,nyqst)/2
        migSIG[:, nyqst - 1] /= 2

        # MATLAB: migSIG(:,nyqst+dx) = migSIG(:,nyqst)
        migSIG[:, nyqst + dx - 1] = (
            migSIG[:, nyqst - 1]
        )

        migSIG = migSIG * ncol / nx

        nx = ncol

    # ========================================================
    # 11. FINAL SPATIAL IFFT
    # MATLAB: ifft(migSIG,[],2,'symmetric')
    # ========================================================

    migSIG = np.fft.ifft(
        migSIG, axis=1
    ).real

    # MATLAB: migSIG((1:nt)+ntshift,1:nx)
    migSIG = migSIG[
        ntshift:ntshift + nt,
        :nx
    ]

    # ========================================================
    # 12. GRID COORDINATES
    # ========================================================

    param['x'] = (
        (np.arange(nx) - (nx - 1) / 2)
        * param['pitch']
    )

    param['z'] = (
        (np.arange(nt) + ntshift)
        * c / 2 / param['fs']
    )

    return migSIG, param


# ============================================================
# INTERPOLATION 1: LINEAR INTERPOLATION
# Exact translation of MATLAB interpLIN
# ============================================================

def interpLIN(dx, y, xi, wn):

    siz = y.shape

    yi = np.zeros(
        siz, dtype=np.result_type(y.dtype, np.float64)
    )

    n1 = int(matlab_round(
        wn[0] * (siz[0] - 1) + 1
    ))

    n2 = int(matlab_round(
        wn[1] * (siz[0] - 1) + 1
    ))

    # MATLAB: idx = xi/dx + 1
    idx = xi / dx + 1

    I = idx > (siz[0] - 1)

    idx = idx.copy()
    idx[I] = 1

    idxf = np.floor(idx).astype(int)

    # MATLAB n1:n2 -> Python n1-1:n2
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
# INTERPOLATION 2: P-POINT SINC INTERPOLATION
# Exact translation of MATLAB interpSINC
# ============================================================

def interpSINC(dx, y, xi, p, wn):

    def sinc0(x):
        return 1 + (8 / np.pi - 4) * x**2

    siz = y.shape

    yi = np.zeros(
        siz, dtype=np.complex128
    )

    n1 = int(matlab_round(
        wn[0] * (siz[0] - 1) + 1
    ))

    n2 = int(matlab_round(
        wn[1] * (siz[0] - 1) + 1
    ))

    xi = xi / dx + 1

    # MATLAB round
    idxr = matlab_round(xi).astype(int)

    delta_n = xi - idxr

    SINmat = np.sin(np.pi * delta_n)

    EXPmat = (
        np.cos(np.pi * delta_n)
        - 1j * SINmat
    )

    # MATLAB:
    # mp = [floor(-p/2+1):-1 1:floor(p/2)]

    mp = (
        list(range(int(np.floor(-p / 2 + 1)), 0))
        +
        list(range(1, int(np.floor(p / 2)) + 1))
    )

    start = n1 - 1
    stop = n2

    # MATLAB: for k = 2:siz(2)/2+1
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

                yi[start:stop, siz[1] - k] += (
                    C * yk2[idxk - 1]
                )

        yi[start:stop, k] = (
            1 / np.pi
            * SINmat[start:stop, k]
            * yi[start:stop, k]
        )

        if k < siz[1] // 2:

            yi[start:stop, siz[1] - k] = (
                1 / np.pi
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
            (4 - 8 / np.pi) ** -0.5
        )

        S = sinc0(delta_nk)

        yi[start:stop, k] += (
            yk1[idxrk - 1] * S
        )

        if k < siz[1] // 2:

            yi[start:stop, siz[1] - k] += (
                yk2[idxrk - 1] * S
            )

        yi[start:stop, k] *= (
            EXPmat[start:stop, k]
        )

        if k < siz[1] // 2:

            yi[start:stop, siz[1] - k] *= (
                EXPmat[start:stop, k]
            )

    return yi

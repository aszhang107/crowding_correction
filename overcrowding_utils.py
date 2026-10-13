import numpy as np
from fle_2d import FLEBasis2D
import tqdm
from scipy.signal import fftconvolve, oaconvolve

def estimate_mean_naive(src, batch_size = 2000):
    img_size = src.L
    mean = np.zeros((img_size, img_size), dtype = np.complex128)
    denom = np.zeros((img_size, img_size), dtype = np.complex128)
    num_batches = src.n // batch_size + 1
    evaled_ctfs = np.array([ctf.evaluate_grid(img_size, pixel_size = src.pixel_size) for ctf in src.unique_filters])
    for batch_num in tqdm.tqdm(range(num_batches)):
        lb = batch_num * batch_size
        ub = min((batch_num + 1) * batch_size, src.n)
        images = src.images[lb:ub].asnumpy() 
        filter_indices = src.filter_indices[lb:ub]
        ft_images = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(images))) 
        mean += np.sum(evaled_ctfs[filter_indices] * ft_images, axis = 0)
        denom += np.sum(evaled_ctfs[filter_indices]**2, axis = 0)
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(mean / denom)))

def estimate_covar_naive(src, noise_mat,noise_psds=None, batch_size = 1000):
    img_size = src.L
    if noise_psds is None:
        noise_psds = np.zeros((len(src.unique_filters), img_size, img_size))
    covar = np.zeros((img_size * img_size, img_size * img_size), dtype = np.complex128)
    denom = np.zeros(covar.shape, dtype = np.complex128)
    num_batches = src.n // batch_size + 1
    evaled_ctfs = np.array([ctf.evaluate_grid(img_size, pixel_size = src.pixel_size) for ctf in src.unique_filters])    
    for batch_num in tqdm.tqdm(range(num_batches)):
        lb = batch_num * batch_size
        ub = min((batch_num + 1) * batch_size, src.n)
        images = src.images[lb:ub].asnumpy() 
        ft_images = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(images))).reshape(-1, img_size**2)
        filter_indices = src.filter_indices[lb:ub]
        psds = noise_psds[filter_indices].reshape(-1, img_size**2)
        H = evaled_ctfs[filter_indices].reshape(-1, img_size**2)  
        X = H * ft_images.reshape(-1, img_size**2)  
        noise_correction = (
            src.L**2
            * noise_mat.multiply(
                (H * np.sqrt(psds)).T @ (H * np.sqrt(psds))
            )
        ).toarray()
        covar += (X.T @ X - noise_correction)
        denom += (H**2).T @ H**2
    A = (covar / denom).reshape(img_size, img_size, img_size, img_size)
    A = np.fft.fftshift(A, axes=(0, 1))
    A = np.fft.ifft2(A, axes=(0, 1))
    A = np.fft.ifftshift(A, axes=(0, 1))
    A = np.fft.fftshift(A, axes=(2, 3))
    A = np.fft.ifft2(A, axes=(2, 3))
    A = np.fft.ifftshift(A, axes=(2, 3))

    return A.reshape(img_size * img_size, img_size * img_size)

def estimate_mean_ctfless(src, batch_size = 2000):
    img_size = src.L
    mean = np.zeros((img_size, img_size), dtype = np.complex128)
    denom = np.zeros((img_size, img_size), dtype = np.complex128)
    num_batches = src.n // batch_size + 1
    for batch_num in tqdm.tqdm(range(num_batches)):
        lb = batch_num * batch_size
        ub = min((batch_num + 1) * batch_size, src.n)
        images = src.images[lb:ub].asnumpy() 
        mean += np.sum(images, axis = 0)
    return mean / src.n

def estimate_covar_ctfless(src, batch_size = 400):
    img_size = src.L
    covar = np.zeros((img_size * img_size, img_size * img_size), dtype = np.float64)
    num_batches = src.n // batch_size + 1
    for batch_num in tqdm.tqdm(range(num_batches)):
        lb = batch_num * batch_size
        ub = min((batch_num + 1) * batch_size, src.n)
        images = src.images[lb:ub].asnumpy().reshape(-1, img_size**2)
        covar += images.T @ images
    return covar / src.n

def conv_diag(cov, d):
    N = cov.shape[0]
    M = 2 * N
    full_size = 3 * N - 1
    lo = N
    hi = full_size - N + 1
    nout = hi - lo
    out = np.zeros(
        (nout, nout, nout, nout),
        dtype=np.result_type(cov, d)
    )
    for dx in range(-(N - 1), N):
        for dy in range(-(N - 1), N):
            C = np.zeros((N, N), dtype=cov.dtype)
            a0 = max(0, -dx)
            a1 = min(N, N - dx)
            b0 = max(0, -dy)
            b1 = min(N, N - dy)
            if a0 < a1 and b0 < b1:
                a = np.arange(a0, a1)
                b = np.arange(b0, b1)
                C[
                    a[:, None],
                    b[None, :]
                ] = cov[
                    a[:, None],
                    b[None, :],
                    (a + dx)[:, None],
                    (b + dy)[None, :]
                ]
            Z = fftconvolve(C, d, mode="full")
            i0 = max(lo, lo - dx)
            i1 = min(hi, hi - dx)
            j0 = max(lo, lo - dy)
            j1 = min(hi, hi - dy)
            if i0 >= i1 or j0 >= j1:
                continue
            ii = np.arange(i0, i1)
            jj = np.arange(j0, j1)
            oi = ii - lo
            oj = jj - lo
            ok = ii + dx - lo
            ol = jj + dy - lo
            out[
                oi[:, None],
                oj[None, :],
                ok[:, None],
                ol[None, :]
            ] = Z[
                ii[:, None],
                jj[None, :]
            ]
    return out

def make_p_eff(p_hi, psd_size, m):

    n = psd_size
    center = m*n

    peff = np.zeros((2*n, 2*n))

    ax = np.arange(p_hi.shape[0]) - center
    ay = np.arange(p_hi.shape[1]) - center

    for r in range(m):
        for s in range(m):

            cx = np.floor_divide(ax + r, m)
            cy = np.floor_divide(ay + s, m)

            ix = cx + n
            iy = cy + n

            for i in range(len(ax)):
                for j in range(len(ay)):
                    if 0 <= ix[i] < 2*n and 0 <= iy[j] < 2*n:
                        peff[ix[i], iy[j]] += p_hi[i,j] / m**2

    return peff / m**2

def estimate_psd(micrograph, centers, psd_size):

    M = micrograph.shape[0]
    mask = np.ones((M, M), dtype=np.float64)

    half = psd_size // 2

    for row, col in centers:
        r0 = max(0, row - half)
        r1 = min(M, row + half)

        c0 = max(0, col - half)
        c1 = min(M, col + half)

        mask[r0:r1, c0:c1] = 0.0
    fft_shape = (2 * M - 1, 2 * M - 1)
    x = micrograph * mask
    Fx = np.fft.fft2(x, s=fft_shape)
    numerator = np.fft.ifft2(
        Fx * np.conj(Fx)
    ).real
    Fm = np.fft.fft2(mask, s=fft_shape)
    denominator = np.fft.ifft2(
        Fm * np.conj(Fm)
    ).real # Need to deconvolve with the FT mask...
    numerator = np.fft.fftshift(numerator)
    denominator = np.fft.fftshift(denominator)
    h = psd_size // 2
    center = M - 1
    numerator = numerator[
        center - h:center + h,
        center - h:center + h
    ]
    denominator = denominator[
        center - h:center + h,
        center - h:center + h
    ]
    acf = np.zeros_like(numerator)
    np.divide(
        numerator,
        denominator,
        out=acf,
        where=denominator > 0
    )
    psd = np.fft.ifftshift(
        np.fft.fft2(
            np.fft.fftshift(acf)
        )
    )
    fle = FLEBasis2D(psd_size, psd_size, 1e-6)
    coeffs = fle.expand(psd.real)
    coeffs[fle.ns != 0] = 0
    return fle.evaluate(coeffs)


def estimate_psd_old(micrograph, centers, psd_size):
    half = psd_size // 2
    M = micrograph.shape[0]
    excluded = np.zeros((M, M), dtype=bool)
    for row, col in centers:
        r0 = max(0, row - half)
        r1 = min(M, row + half)
        c0 = max(0, col - half)
        c1 = min(M, col + half)
        excluded[r0:r1, c0:c1] = True
    lags = np.arange(-psd_size//2, psd_size//2)
    acf = np.zeros((psd_size, psd_size), dtype = np.float64)
    for i, dr in enumerate(lags):
        for j, dc in enumerate(lags):
            r0 = max(0, -dr)
            r1 = min(M, M - dr)
            c0 = max(0, -dc)
            c1 = min(M, M - dc)
            a1 = micrograph[r0:r1, c0:c1]
            a2 = micrograph[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            e1 = excluded[r0:r1, c0:c1]
            e2 = excluded[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            valid = ~(e1 | e2)
            if np.any(valid):
                acf[i, j] = np.mean(a1[valid] * a2[valid])
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(acf)))
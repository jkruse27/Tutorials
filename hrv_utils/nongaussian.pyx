# lambda_fast.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from __future__ import annotations
 
import math
from scipy.signal import savgol_filter, savgol_coeffs
from scipy.special import gammaln
from joblib import Parallel, delayed
import numpy as np
cimport numpy as cnp
import cython
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.math cimport log, fabs, pow, M_PI, lgamma, isnan, sqrt, isinf, exp
from libc.time cimport time
from scipy.linalg import pinv

cdef double EULER_MASCHERONI = 0.57721566490153286060
cdef double M_CONST = (EULER_MASCHERONI + log(2.0)) / 2.0
cdef double LOG_PI_DIV_2 = log(M_PI) / 2.0
_LOG2 = math.log(2.0)
_LOG_PI_DIV_2 = math.log(math.pi) / 2.0
_M_CONST = (math.log(2.0) + math.log(math.pi)) / 2.0 - math.log(2.0) / 2.0
cnp.import_array()

def sgolay(int p, int n, int m=0, double ts=1.0):
    """
    Computes the Savitzky-Golay differentiation filters.
    Replicates R's signal::sgolay logic exactly.
    
    Returns:
        F (np.ndarray): n x n matrix of filter coefficients.
    """
    if n % 2 != 1:
        raise ValueError("sgolay: window length 'n' must be odd.")
    if p >= n:
        raise ValueError("sgolay: polynomial order 'p' must be less than 'n'.")

    k = n // 2
    ind = np.arange(-k, k + 1, dtype=np.float64)
    powers = np.arange(0, p + 1, dtype=np.float64)
    V = ind[:, None] ** powers[None, :]

    B = pinv(V).T 

    F = V @ B.T

    if m > 0:
        F = np.zeros((n, n), dtype=np.float64)
        for row_idx in range(n):
            s = row_idx - k

            basis_vals = np.zeros(p + 1)
            for j in range(p + 1):
                if j < m:
                    basis_vals[j] = 0
                else:
                    fact = 1.0
                    for l in range(m):
                        fact *= (j - l)
                    if s == 0 and (j - m) == 0:
                        term = 1.0
                    else:
                        term = pow(s, j - m)
                    basis_vals[j] = fact * term

            F[row_idx, :] = basis_vals @ B.T

    if m > 0 and ts != 1.0:
        F /= (ts ** m)
        
    return F

def sgolayfilt(double[:] x, int p=3, int n=0, int m=0, double ts=1.0):
    """
    Cython implementation of R's sgolayfilt.
    
    Parameters:
        x (double[:]): Input data vector
        p (int): Polynomial order
        n (int): Window length (must be odd). If 0, defaults to p + 3 - (p%2)
        m (int): Derivative order
        ts (double): Time scaling factor
    """
    if n == 0:
        n = p + 3 - (p % 2)
    
    if n % 2 == 0:
        raise ValueError("n must be odd")
        
    cdef Py_ssize_t length_x = x.shape[0]
    cdef Py_ssize_t k = n // 2
    cdef Py_ssize_t i, j, r

    cdef double[:, ::1] F = sgolay(p, n, m, ts)

    cdef double[:] output = np.zeros(length_x, dtype=np.float64)

    cdef double val
    for i in range(k):
        val = 0.0
        for j in range(n):
            val += F[i, j] * x[j]
        output[i] = val

    cdef double[:] center_coeffs = np.zeros(n, dtype=np.float64)
    for j in range(n):
        center_coeffs[j] = F[k, j]

    cdef Py_ssize_t start_idx
    for i in range(k, length_x - k):
        val = 0.0
        start_idx = i - k
        for j in range(n):
            val += center_coeffs[j] * x[start_idx + j]
        output[i] = val

    
    cdef Py_ssize_t row_in_F
    cdef Py_ssize_t x_start = length_x - n

    for r in range(k):
        row_in_F = k + 1 + r
        
        val = 0.0
        for j in range(n):
            val += F[row_in_F, j] * x[x_start + j]
            
        output[length_x - k + r] = val
        
    return output


def nongaussian_analysis(double[:] series, long[:] scales, double q=0.25, int m=3):
    """
    Performs the full non-Gaussian analysis over a range of scales.
    
    Parameters:
        series: The input time series (1D array)
        scales: Array of window lengths (must be odd integers)
        q: Moment order (default 0.25)
        m: Savitzky-Golay polynomial order (default 3)
        
    Returns:
        (nongaussianity, curves): Tuple of lists
    """
    cdef Py_ssize_t s_idx, scale
    cdef Py_ssize_t n_scales = scales.shape[0]

    nongaussianity = []
    curves = []

    series_np = np.asarray(series)
    y = np.cumsum(series_np - np.nanmean(series_np))

    for s_idx in range(n_scales):
        scale = scales[s_idx]
        if scale % 2 == 0:
            scale += 1
        y_sg = sgolayfilt(y, m, int(scale))
        
        y_detrend = y - y_sg

        dy_raw = y_detrend[scale:] - y_detrend[:-scale]
        dy_mean = np.nanmean(dy_raw)
        dy_std = np.nanstd(dy_raw, ddof=1)
        
        if dy_std == 0:
            dy_std = 1.0
            
        dy = (dy_raw - dy_mean) / dy_std

        lam_sq = _calculate_single_lambda(dy, q, 1e-3)
        nongaussianity.append(lam_sq)
        curves.append(dy)
        
    return nongaussianity, curves


def nongaussian_index(double[:] x, double q):
    """
    Computes the nongaussian index (lambda^2) for a single series.
    Direct replacement for the Python function.
    """
    return _calculate_single_lambda(x, q, 1e-3)


def estimate_lambda_sq(double[:] x, double[:] q, double tol=1e-3):
    """
    Vectorized version: Computes lambda^2 for multiple q values.
    """
    cdef Py_ssize_t n_q = q.shape[0]
    cdef Py_ssize_t i
    cdef double[:] out_view
    out = np.empty(n_q, dtype=np.float64)
    out_view = out

    for i in range(n_q):
        out_view[i] = _calculate_single_lambda(x, q[i], tol)
            
    return out


cdef double _calculate_single_lambda(double[:] x, double q, double tol) nogil:
    """
    Dispatch logic for a single q value.
    """
    if fabs(q - 0.0) <= tol:
        return _calc_case_q0(x)
    elif fabs(q - 2.0) <= tol:
        return _calc_case_q2(x)
    else:
        return _calc_case_general(x, q)


cdef double _calc_case_q0(double[:] x) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t N = x.shape[0]
    cdef double val
    cdef double sum_lax = 0.0
    cdef Py_ssize_t count = 0
    
    for i in range(N):
        val = x[i]
        if not isnan(val) and not isinf(val):
            sum_lax += log(fabs(val))
            count += 1
            
    if count == 0: return 0.0
    return -(sum_lax / count) - M_CONST


cdef double _calc_case_q2(double[:] x) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t N = x.shape[0]
    cdef double val, val_sq, lax
    cdef double sum_term = 0.0
    cdef Py_ssize_t count = 0
    
    for i in range(N):
        val = x[i]
        if not isnan(val) and not isinf(val):
            val_sq = val * val
            lax = log(fabs(val))
            sum_term += val_sq * lax
            count += 1

    if count == 0: return 0.0
    return (sum_term / count) + M_CONST - 1.0


cdef double _calc_case_general(double[:] x, double q) nogil:
    cdef Py_ssize_t i
    cdef Py_ssize_t N = x.shape[0]
    cdef double val
    cdef double sum_pow = 0.0
    cdef Py_ssize_t count = 0
    
    for i in range(N):
        val = x[i]
        if not isnan(val) and not isinf(val):
            sum_pow += pow(fabs(val), q)
            count += 1
            
    if count == 0: return 0.0
    
    cdef double m_absq = sum_pow / count
    cdef double c1 = -(log(2.0) * q / 2.0)
    cdef double c2 = -lgamma((q + 1.0) / 2.0)
    cdef double pre_factor = 2.0 / (q * (q - 2.0))
    cdef double term = LOG_PI_DIV_2 + log(m_absq) + c1 + c2
    
    return pre_factor * term


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _renormalize_inplace(double[:] x, double new_std, double new_mu) noexcept nogil:
    """
    C-level helper to renormalize an array in-place.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t n = x.shape[0]
    cdef double sum_x = 0.0
    cdef double sum_sq_diff = 0.0
    cdef double mean, std, val
    
    # 1. Calculate Mean
    for i in range(n):
        sum_x += x[i]
    mean = sum_x / n
    
    # 2. Calculate Standard Deviation
    for i in range(n):
        sum_sq_diff += (x[i] - mean) * (x[i] - mean)
    
    # Handle edge case where array is constant or empty
    if n > 0 and sum_sq_diff > 0:
        std = sqrt(sum_sq_diff / n)
    else:
        std = 1.0 

    # 3. Transform in-place
    # Formula: (x - mean) / std * new_std - new_mu
    cdef double factor = new_std / std
    
    for i in range(n):
        x[i] = (x[i] - mean) * factor - new_mu

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def renormalize(cnp.ndarray[cnp.double_t, ndim=1] x, double std, double mu):
    """
    Python-exposed wrapper if you need to call renormalize independently.
    Returns a new array to match original Python behavior.
    """
    cdef cnp.ndarray[cnp.double_t, ndim=1] x_copy = x.copy()
    _renormalize_inplace(x_copy, std, mu)
    return x_copy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def generate_nongaussian(
    int m,
    double lmd2,
    int n0
):
    """
    Cython optimized generation of non-gaussian series.
    """
    cdef:
        double lmd2e = lmd2 / m
        double lmd0 = sqrt(lmd2e)
        Py_ssize_t total_len = n0 * (1 << m)  # n0 * 2^m
        Py_ssize_t i, j, k
        Py_ssize_t layer_len, repeat_count
        
        # We allocate the accumulator ONCE. 
        # Original code summed a list of arrays; we sum as we go.
        cnp.ndarray[cnp.double_t, ndim=1] accumulator = np.zeros(total_len, dtype=np.float64)
        double[:] acc_view = accumulator
        
        # Temporary buffer for the current layer's noise
        cnp.ndarray[cnp.double_t, ndim=1] noise_layer
        double[:] noise_view
        double val

    # Loop through cascading levels
    for i in range(m):
        layer_len = n0 * (1 << i)
        repeat_count = 1 << (m - i)
        
        # Generate random normal numbers (Numpy is fast enough for generation)
        noise_layer = np.random.normal(size=layer_len)
        noise_view = noise_layer
        
        # 1. Renormalize this small layer in-place (no GIL)
        with nogil:
            _renormalize_inplace(noise_view, lmd0, lmd2e)
        
            # 2. Add to accumulator with repetition (simulating np.repeat)
            # This avoids creating the massive expanded array in memory
            for j in range(layer_len):
                val = noise_view[j]
                for k in range(repeat_count):
                    # We map the smaller layer onto the total length
                    # equivalent to: np.repeat(noise, repeat_count)
                    acc_view[j * repeat_count + k] += val

    # Final transformation: exp(sum) * new_noise
    # We can do this efficiently in Numpy or a simple loop. 
    # Since we need to generate new noise anyway, Numpy is clean here.
    cdef cnp.ndarray[cnp.double_t, ndim=1] final_noise = np.random.normal(size=total_len)
    
    # In-place exponentiation and multiplication
    # x = exp(x)
    np.exp(accumulator, out=accumulator)
    # x = x * final_noise
    np.multiply(accumulator, final_noise, out=accumulator)

    return accumulator


def _lambda_batch(dy: np.ndarray, q: float, tol: float = 1e-3) -> np.ndarray:
    """
    Compute lambda^2 for a batch of normalised increment series.
 
    Parameters
    ----------
    dy : ndarray, shape (n_surrogates, n_samples)
        Normalised increments.  NaN/Inf values are ignored (nanmean).
    q  : float
        Moment order.
 
    Returns
    -------
    lam_sq : ndarray, shape (n_surrogates,)
    """
    if abs(q) <= tol:
        # q = 0 case:  -(E[log|x|] + M_CONST)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_abs = np.log(np.abs(dy))
        log_abs[~np.isfinite(log_abs)] = np.nan
        return -(np.nanmean(log_abs, axis=1) + _M_CONST)
 
    if abs(q - 2.0) <= tol:
        # q = 2 case:  E[x^2 * log|x|] + M_CONST - 1
        with np.errstate(divide="ignore", invalid="ignore"):
            log_abs = np.log(np.abs(dy))
        vals = dy ** 2 * log_abs
        vals[~np.isfinite(vals)] = np.nan
        return np.nanmean(vals, axis=1) + _M_CONST - 1.0
 
    # General case
    with np.errstate(invalid="ignore"):
        abs_pow = np.abs(dy) ** q          # (n_surrogates, n_samples)
    abs_pow[~np.isfinite(abs_pow)] = np.nan
    m_absq = np.nanmean(abs_pow, axis=1)   # (n_surrogates,)
 
    c1 = -(_LOG2 * q / 2.0)
    c2 = -gammaln((q + 1.0) / 2.0)
    pre_factor = 2.0 / (q * (q - 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        term = _LOG_PI_DIV_2 + np.log(m_absq) + c1 + c2
    return pre_factor * term
 
 
# ── Per-scale worker (runs in parallel) ──────────────────────────────────── #
 
def _process_scale(
    Y_cumsum: np.ndarray,   # (n_surrogates, n)  float64
    scale: int,
    q: float,
    m: int,
    sg_coeffs: np.ndarray,  # precomputed for this scale
) -> tuple[np.ndarray, np.ndarray]:
    """
    Process one scale for the entire surrogate batch.
 
    Returns
    -------
    lam_sq : ndarray, shape (n_surrogates,)
    dy     : ndarray, shape (n_surrogates, n - scale)   normalised increments
    """
    # --- Savitzky-Golay detrend (all surrogates at once, axis=1) ----------- #
    # scipy.signal.savgol_filter natively handles nd arrays along any axis.
    # Passing precomputed coefficients avoids recomputing them per call.
    y_sg = savgol_filter(Y_cumsum, window_length=scale, polyorder=m, axis=1)
    y_detrend = Y_cumsum - y_sg                          # (n_surrogates, n)
 
    # --- Increments --------------------------------------------------------- #
    dy_raw = y_detrend[:, scale:] - y_detrend[:, :-scale]  # (n_surrogates, n-scale)
 
    # --- Standardise (per surrogate) --------------------------------------- #
    dy_mean = np.nanmean(dy_raw, axis=1, keepdims=True)
    dy_std  = np.nanstd(dy_raw,  axis=1, keepdims=True, ddof=1)
    dy_std  = np.where(dy_std == 0, 1.0, dy_std)
    dy = (dy_raw - dy_mean) / dy_std
 
    # --- Lambda ------------------------------------------------------------ #
    lam_sq = _lambda_batch(dy, q)
 
    return lam_sq, dy
 
 
# ── Public API ────────────────────────────────────────────────────────────── #
 
def nongaussian_analysis_batch(
    surrogates: np.ndarray,
    scales: np.ndarray,
    q: float = 0.25,
    m: int = 3,
    n_jobs: int = 1,
    return_curves: bool = False,
) -> tuple[np.ndarray, list | None]:
    """
    Batch non-Gaussianity analysis over a range of scales.
 
    Processes all surrogates simultaneously for each scale, which is far more
    efficient than calling the per-surrogate function in a loop.
 
    Parameters
    ----------
    surrogates : ndarray, shape (n_surrogates, n)
        Surrogate (or original) time series, one per row.
    scales : array_like of int
        Window lengths for detrending.  Odd values are used; even values are
        incremented by 1 (matches original behaviour).
    q : float, default 0.25
        Moment order for lambda^2.
    m : int, default 3
        Savitzky-Golay polynomial order.
    n_jobs : int, default 1
        Number of parallel workers for the scale loop.  -1 uses all cores.
        Set to 1 to disable parallelism (safe for nested parallel calls).
    return_curves : bool, default False
        If True, also return the normalised increment arrays.  These are
        ragged (each scale gives a different length), so they are returned
        as a list of (n_surrogates, n - scale) arrays, one per scale.
        Disable if not needed to avoid the memory cost.
 
    Returns
    -------
    nongaussianity : ndarray, shape (n_scales, n_surrogates)
        lambda^2 values.  Row i corresponds to scales[i].
    curves : list of ndarray or None
        Normalised increments per scale (only when return_curves=True).
    """
    surrogates = np.asarray(surrogates, dtype=np.float64)
    if surrogates.ndim == 1:
        surrogates = surrogates[np.newaxis, :]   # single signal → batch of 1
    n_surrogates, n = surrogates.shape
 
    scales = np.asarray(scales, dtype=np.intp)
    # Enforce odd window lengths
    scales = np.where(scales % 2 == 0, scales + 1, scales)
    n_scales = scales.size
 
    # ── Cumulative sum (profile), mean-centred — done once for all scales ── #
    means = np.nanmean(surrogates, axis=1, keepdims=True)  # (n_surrogates, 1)
    Y_cumsum = np.cumsum(surrogates - means, axis=1)       # (n_surrogates, n)
 
    # ── Precompute SG coefficients per unique scale ─────────────────────── #
    # savgol_coeffs is pure Python/C and cheap; hoisting it avoids repeated
    # work inside joblib workers.
    unique_scales = np.unique(scales)
    sg_coeff_map = {
        int(s): savgol_coeffs(int(s), m) for s in unique_scales
    }
 
    # ── Parallel scale loop ─────────────────────────────────────────────── #
    def _job(scale):
        return _process_scale(
            Y_cumsum, int(scale), q, m, sg_coeff_map[int(scale)]
        )
 
    if n_jobs == 1:
        results = [_job(s) for s in scales]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_job)(s) for s in scales
        )
 
    # ── Assemble outputs ─────────────────────────────────────────────────── #
    nongaussianity = np.stack([r[0] for r in results], axis=0)  # (n_scales, n_surrogates)
    curves = [r[1] for r in results] if return_curves else None
 
    return nongaussianity, curves
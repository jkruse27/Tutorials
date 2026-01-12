# lambda_fast.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.math cimport log, fabs, pow, M_PI, lgamma, isnan, sqrt, isinf, exp, cos, sin
from libc.time cimport time
from scipy.linalg import pinv

cdef double EULER_MASCHERONI = 0.57721566490153286060
cdef double M_CONST = (EULER_MASCHERONI + log(2.0)) / 2.0
cdef double LOG_PI_DIV_2 = log(M_PI) / 2.0


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


def generate_cascade_series(double lambda2_total, int n_level, int n_base):
    cdef double lambda2_per_level = lambda2_total / n_level
    cdef double sigma_logw = sqrt(lambda2_per_level)
    cdef double mu = -lambda2_per_level / 2.0
    cdef double two_pi = 2.0 * M_PI
    srand(time(NULL))
    cdef cnp.ndarray[cnp.double_t, ndim=1] weights = np.ones(n_base, dtype=np.float64)
    cdef cnp.ndarray[cnp.double_t, ndim=1] new_weights

    cdef double[:] w_view
    cdef double[:] new_w_view

    cdef int lev, i, n_parent, idx_l, idx_r
    cdef double u1, u2, z1, z2, mag, w_val

    for lev in range(n_level):
        n_parent = weights.shape[0]
        new_weights = np.empty(n_parent * 2, dtype=np.float64)
        w_view = weights
        new_w_view = new_weights

        for i in range(n_parent):
            u1 = (rand() + 1.0) / (RAND_MAX + 2.0)
            u2 = (rand() + 1.0) / (RAND_MAX + 2.0)

            mag = sqrt(-2.0 * log(u1))
            z1 = mag * cos(two_pi * u2)
            z2 = mag * sin(two_pi * u2)

            w_val = w_view[i]
            new_w_view[2 * i] = w_val * exp(mu + (sigma_logw * z1))
            new_w_view[2 * i + 1] = w_val * exp(mu + (sigma_logw * z2))

        weights = new_weights

    cdef int n_final = weights.shape[0]
    w_view = weights
    
    for i in range(0, n_final, 2):
        u1 = (rand() + 1.0) / (RAND_MAX + 2.0)
        u2 = (rand() + 1.0) / (RAND_MAX + 2.0)
        
        mag = sqrt(-2.0 * log(u1))
        z1 = mag * cos(two_pi * u2)
        z2 = mag * sin(two_pi * u2)
        
        w_view[i] *= z1

        if i + 1 < n_final:
            w_view[i + 1] *= z2
            
    return weights
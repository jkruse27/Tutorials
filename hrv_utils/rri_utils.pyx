# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, realloc, qsort
# CHANGED: Added fabs here, removed abs from stdlib
from libc.math cimport isnan, NAN, floor, round, fabs 

# ==========================================
# 1. Structs & Constants
# ==========================================

cdef struct Point:
    double time
    double rri

# ==========================================
# 2. C Helper Functions
# ==========================================

cdef int compare_points(const void *a, const void *b) noexcept nogil:
    cdef double t_a = (<Point*>a).time
    cdef double t_b = (<Point*>b).time
    if t_a < t_b: return -1
    elif t_a > t_b: return 1
    return 0

cdef int compare_doubles(const void *a, const void *b) noexcept nogil:
    cdef double d_a = (<double*>a)[0]
    cdef double d_b = (<double*>b)[0]
    if d_a < d_b: return -1
    elif d_a > d_b: return 1
    return 0

cdef inline void _sort_inplace(double* arr, int n) noexcept nogil:
    cdef int i, j
    cdef double key
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

cdef double calc_median(double* arr, int n) noexcept nogil:
    if n == 0: return 0.0
    qsort(arr, n, sizeof(double), compare_doubles)
    if n % 2 == 1:
        return arr[n // 2]
    else:
        return 0.5 * (arr[(n // 2) - 1] + arr[n // 2])

cdef double* c_moving_median(double* x, int N, int n_win) noexcept nogil:
    cdef double* y = <double*>malloc(N * sizeof(double))
    if y == NULL: return NULL
    
    if n_win > 100: n_win = 100 

    cdef int k = (n_win - 1) // 2
    cdef int i, j, count, i_left, i_right, mid_idx
    cdef double val
    cdef double buffer[100] 

    for i in range(N):
        i_left = i - k
        if i_left < 0: i_left = 0
        i_right = i + k
        if i_right >= N: i_right = N - 1
        
        count = 0
        for j in range(i_left, i_right + 1):
            val = x[j]
            if not isnan(val):
                buffer[count] = val
                count += 1
        
        if count == 0:
            y[i] = NAN
        else:
            _sort_inplace(buffer, count)
            mid_idx = count // 2
            if count % 2 == 1:
                y[i] = buffer[mid_idx]
            else:
                y[i] = (buffer[mid_idx - 1] + buffer[mid_idx]) * 0.5
    return y

# ==========================================
# 3. Main Python-Exposed Functions
# ==========================================

def clean_rri_signal(np.ndarray[double, ndim=1] rri_in, 
                     np.ndarray[double, ndim=1] time_in,
                     int n_med=9, 
                     double rri_max=1600.0, 
                     double rri_min=333.0, 
                     double rri_diff=200.0):
    cdef int N0 = rri_in.shape[0]
    if N0 < 2:
        raise ValueError("RRI length must be >= 2")

    cdef double* c_rri = <double*>malloc(N0 * sizeof(double))
    cdef double* c_time = <double*>malloc(N0 * sizeof(double))
    cdef int i
    
    for i in range(N0):
        c_rri[i] = rri_in[i]
        c_time[i] = time_in[i]

    cdef double* rri_mid = c_moving_median(c_rri, N0, n_med)
    
    cdef int capacity = N0 + (N0 // 2)
    cdef Point* result_buf = <Point*>malloc(capacity * sizeof(Point))
    cdef int result_count = N0 
    
    for i in range(N0):
        result_buf[i].time = c_time[i]
        result_buf[i].rri = c_rri[i]

    cdef double ratio_raw
    cdef int ratio_int
    cdef double tmp_val
    
    for i in range(N0):
        if not isnan(rri_mid[i]) and rri_mid[i] != 0 and not isnan(result_buf[i].rri):
            ratio_raw = result_buf[i].rri / rri_mid[i]
            ratio_int = <int>round(ratio_raw)
        else:
            ratio_int = 1 

        if ratio_int >= 2:
            result_buf[i].rri = result_buf[i].rri / ratio_int
            
            if result_count + (ratio_int - 1) >= capacity:
                capacity += max(100, ratio_int * 2)
                result_buf = <Point*>realloc(result_buf, capacity * sizeof(Point))
            
            for _ in range(ratio_int - 1):
                result_buf[result_count].time = result_buf[i].time
                result_buf[result_count].rri = result_buf[i].rri
                result_count += 1
        
        elif i >= 1: 
            if (not isnan(result_buf[i-1].rri) and not isnan(result_buf[i].rri) and 
                not isnan(rri_mid[i-1]) and not isnan(rri_mid[i])):
                
                if ((result_buf[i-1].rri > rri_mid[i-1] + rri_diff) and 
                    (result_buf[i].rri < rri_mid[i] - rri_diff)):
                    
                    tmp_val = (result_buf[i-1].rri + result_buf[i].rri) / 2.0
                    result_buf[i-1].rri = tmp_val
                    result_buf[i].rri = tmp_val
                
                elif ((result_buf[i-1].rri < rri_mid[i-1] - rri_diff) and 
                      (result_buf[i].rri < rri_mid[i] - rri_diff)):
                    
                    result_buf[i-1].rri += result_buf[i].rri
                    result_buf[i].rri = NAN 

            if not isnan(rri_mid[i]):
                if rri_mid[i] < rri_min or rri_mid[i] > rri_max:
                    result_buf[i].rri = NAN

    cdef int valid_count = 0
    for i in range(result_count):
        if not isnan(result_buf[i].rri) and not isnan(result_buf[i].time):
            result_buf[valid_count] = result_buf[i]
            valid_count += 1
            
    qsort(result_buf, valid_count, sizeof(Point), compare_points)
    
    out_time = np.zeros(valid_count, dtype=np.float64)
    out_rri = np.zeros(valid_count, dtype=np.float64)
    
    for i in range(valid_count):
        out_time[i] = result_buf[i].time
        out_rri[i] = result_buf[i].rri

    free(c_rri)
    free(c_time)
    free(rri_mid)
    free(result_buf)

    return {'time': out_time, 'rri': out_rri}

def resample_signal(np.ndarray[double, ndim=1] time_in, 
                    np.ndarray[double, ndim=1] rri_in, 
                    double t_sec=0.5):
    cdef int n_in = time_in.shape[0]
    if n_in < 2:
        raise ValueError("Input length too short for resampling")

    cdef Point* unique_buf = <Point*>malloc(n_in * sizeof(Point))
    cdef double* tie_buf = <double*>malloc(n_in * sizeof(double))
    
    cdef int i = 0
    cdef int j = 0
    cdef int unique_count = 0
    cdef int k
    
    while i < n_in:
        j = i
        while j < n_in and time_in[j] == time_in[i]:
            tie_buf[j - i] = rri_in[j]
            j += 1
        
        unique_buf[unique_count].time = time_in[i]
        unique_buf[unique_count].rri = calc_median(tie_buf, j - i)
        unique_count += 1
        i = j

    free(tie_buf)

    if unique_count < 2:
        free(unique_buf)
        raise ValueError("Not enough unique timestamps to resample")

    cdef double dt_hour = t_sec / 3600.0
    cdef double start_time = unique_buf[0].time
    cdef double end_time = unique_buf[unique_count - 1].time
    
    cdef int n_out = <int>floor((end_time - start_time) / dt_hour) + 1
    cdef double* out_time = <double*>malloc(n_out * sizeof(double))
    cdef double* out_rri = <double*>malloc(n_out * sizeof(double))
    
    cdef int idx_in = 0
    cdef double t_target
    cdef double t1, t2, y1, y2, slope
    
    for k in range(n_out):
        t_target = start_time + k * dt_hour
        out_time[k] = t_target
        
        while idx_in < unique_count - 1 and unique_buf[idx_in + 1].time < t_target:
            idx_in += 1
            
        if t_target < unique_buf[0].time or t_target > unique_buf[unique_count-1].time:
            # CHANGED: Use fabs for floating point absolute value
            if fabs(t_target - unique_buf[0].time) < 1e-9:
                out_rri[k] = unique_buf[0].rri
            elif fabs(t_target - unique_buf[unique_count-1].time) < 1e-9:
                out_rri[k] = unique_buf[unique_count-1].rri
            else:
                out_rri[k] = NAN
            continue
            
        t1 = unique_buf[idx_in].time
        y1 = unique_buf[idx_in].rri
        
        if idx_in < unique_count - 1:
            t2 = unique_buf[idx_in + 1].time
            y2 = unique_buf[idx_in + 1].rri
            if t2 != t1:
                slope = (y2 - y1) / (t2 - t1)
                out_rri[k] = y1 + slope * (t_target - t1)
            else:
                out_rri[k] = y1
        else:
            out_rri[k] = y1

    res_time = np.zeros(n_out, dtype=np.float64)
    res_rri = np.zeros(n_out, dtype=np.float64)
    
    for k in range(n_out):
        res_time[k] = out_time[k]
        res_rri[k] = out_rri[k]

    free(unique_buf)
    free(out_time)
    free(out_rri)
    
    return {'time': res_time, 'rri': res_rri}

def get_clean_rr_intervals(np.ndarray[double, ndim=1] rri_raw):
    cdef int n = rri_raw.shape[0]
    if n < 2:
        raise ValueError("Input RRI must have length >= 2")

    cdef np.ndarray[double, ndim=1] time_rel = np.zeros(n, dtype=np.float64)
    cdef double current_t = 0.0
    cdef int i

    for i in range(n):
        current_t += rri_raw[i]
        time_rel[i] = current_t

    cleaned = clean_rri_signal(rri_raw, time_rel)
    return cleaned['rri']

def clean_and_resample_pipeline(np.ndarray[double, ndim=1] rri_raw, 
                                double start_hour, 
                                double t_sec=0.5):
    cdef int n = rri_raw.shape[0]
    cdef np.ndarray[double, ndim=1] time_rri = np.empty(n, dtype=np.float64)
    cdef double current_t = start_hour
    cdef int i
    
    for i in range(n):
        current_t += rri_raw[i] / 3600000.0
        time_rri[i] = current_t
        
    cleaned = clean_rri_signal(rri_raw, time_rri)
    resampled = resample_signal(cleaned['time'], cleaned['rri'], t_sec)
    
    return resampled
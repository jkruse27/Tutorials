import cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, pow

def create_scales(
        min_box,
        max_box,
        ratio = np.exp2(0.125)
    ) -> np.ndarray:
    """Function that creates a list of scales in the specified range."""
    rslen = int(np.log10(max_box / min_box) / np.log10(ratio)) + 1
    
    rs = np.empty((rslen,))
    rs[0] = min_box
    rs[1:] = ratio
    rs = np.cumprod(rs) + 0.5

    rs = rs[rs < max_box]

    return np.unique((rs//2)*2+1).astype(np.int_)


def dma(
    series,
    scales,
    order,
    q = 2,
    integrate = 1
    ):
    if (order == 2):
        scales = scales[scales > 4]
    elif (order == 4):
        scales = scales[scales > 6]
    if(q == 2):
        return np.sqrt(np.asarray(
            compute_dma(series, scales, order, integrate)
            ))
    else:
        return np.asarray(
             compute_dma_q(series, scales, order, integrate, q)
            )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:] compute_dma(
    double[:] series,
    long[:] scales,
    int order,
    int integrate = 1
    ):
    cdef long n, i, n_scales, i_refresh
    cdef double[:] f2
    cdef double[:,:] local_sum

    if(integrate):
        series = np.cumsum(series - np.mean(series))
    
    n = series.shape[0]
    n_scales = scales.shape[0]

    if(order == 0):
        i_refresh = 50000
    elif(order == 2):
        i_refresh = 5000
    else:
        i_refresh = 100
    
    if(i_refresh > n): i_refresh = n

    local_sum = np.zeros((5, <long>(n/i_refresh)+2), dtype=np.float64)
    f2 = np.empty(n_scales)

    with nogil:
        for i in range(0, n_scales):
            f2[i] = estimate_f2(series, scales, i, order, i_refresh, local_sum)

    return f2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double[:] compute_dma_q(
    double[:] series,
    long[:] scales,
    int order,
    int integrate = 1,
    float q = 2
    ):
    cdef long n, i, n_scales, i_refresh
    cdef double[:] f2
    cdef double[:,:] local_sum

    if(integrate):
        series = np.cumsum(series - np.mean(series))
    
    n = series.shape[0]
    n_scales = scales.shape[0]

    if(order == 0):
        i_refresh = 50000
    elif(order == 2):
        i_refresh = 5000
    else:
        i_refresh = 100
    
    if(i_refresh > n): i_refresh = n

    local_sum = np.zeros((5, <long>(n/i_refresh)+2), dtype=np.float64)
    f2 = np.empty(n_scales)

    with nogil:
        for i in range(0, n_scales):
            f2[i] = estimate_f2_q(series, scales, i, order, i_refresh, local_sum, q)

    return f2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int cmat(double k, int order, double* c) nogil:
    cdef double den, k2, k3, k4, k5
    k2 = k*k
    
    if(order == 0):
        c[0] = 1.0/(2.0*k+1.0)
    elif(order == 2):
        k3 = k2*k
        den = 8.0*k3 + 12.0*k2 - 2.0*k - 3.0
        c[0] = (9.0*k2 + 9.0*k - 3.0)/den
        c[1] = -15.0/den
    elif(order == 4):
        k3 = k2*k
        k4 = k3*k
        k5 = k4*k
        den = 180.0 + 72.0*k - 800.0*k2 - 320.0*k3 + 320.0*k4 + 128.0*k5
        c[0] = (180.0 - 750.0*k - 525.0*k2 + 450.0*k3 + 225.0*k4)/den
        c[1] = (1575.0 - 1050.0*k - 1050.0*k2)/den
        c[2] = 945.0/den
    else:
        return -1

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double estimate_f2(
    double[:] series, 
    long[:] scales, 
    long index, 
    int order, 
    long i_refresh,
    double[:,:] local_sums
    ) nogil:
    cdef double c[3]
    cdef long n, i, i0, ic, j, itemp, kb, iloc, k
    cdef double f2, a0, y00, y0i, y10, y1i, y0ii, y1ii, y0iii, y0iv
    cdef double temp, temp1, temp2, nn1, nn2, nn3, nn4, y1iii, y1iv
    cdef long scale = (scales[index])
    
    n = series.shape[0]
    f2 = 0
    k = <long> ((scale-1)/2)
    ic = k 

    if(order == 0):
        y10=0
        for i in range(0, scale):
             y10 += series[i]

        cmat(<double> k, order, c)
        a0 = c[0]*y10
        temp = (series[ic] - a0)
        f2 = temp*temp
        itemp = n - scale

        for i in range(0, itemp):
            if((i+1) % i_refresh == 0):
                y10=0
                for j in range(1, scale+1):
                    if(i+j < n):
                        y10 += series[i+j]
            else:
                y10 += series[i+scale] - series[i]

            a0 = c[0]*y10
            temp = (series[i + ic + 1] - a0)
            f2 += temp*temp

        f2 = f2/<double>(n)

    elif(order == 2):
        iloc = 0
        if(index != 0):
            kb = <long> ((scales[index-1]-1)/2)
        else:
            kb = 1
            
        nn1 = <double>(kb-k)
        nn2 = nn1*nn1

        if(index == 0):
            i0 = 0
            y10 = 0; y1i = 0; y1ii = 0
        else:
            i0 = <long> (scales[index-1])
            y10 = local_sums[0][iloc]
            y1i = local_sums[1][iloc] + local_sums[0][iloc]*nn1
            y1ii = local_sums[2][iloc] + 2*local_sums[1][iloc]*nn1 + local_sums[0][iloc]*nn2

        for i in range(i0, scale):
            y10 += series[i]
            temp1 = series[i]*<double>(i-k)
            y1i += temp1
            temp2 = temp1*<double>(i-k)
            y1ii += temp2

        local_sums[0][iloc] = y10
        local_sums[1][iloc] = y1i
        local_sums[2][iloc] = y1ii

        cmat(<double>k, order, c)
        a0 = c[0]*y10 + c[1]*y1ii
        temp1 = (series[ic] - a0)
        f2 += temp1*temp1
        itemp = n - scale

        for i in range(0, itemp):
            if((i+1) % i_refresh == 0):
                iloc += 1
                if(index == 0):
                    i0 = 0
                    y10 = 0; y1i = 0; y1ii = 0
                else:
                    i0 = scales[index-1]
                    y10 = local_sums[0][iloc]
                    y1i = local_sums[1][iloc] + local_sums[0][iloc]*nn1
                    y1ii = local_sums[2][iloc] + 2*local_sums[1][iloc]*nn1 + local_sums[0][iloc]*nn2

                for j in range(i0, scale):
                    if(i + 1 + j < n):
                        y10 += series[i+1+j]
                        temp1 = series[i+1+j]*<double>(j-k)
                        y1i += temp1
                        temp2 = temp1*<double>(j-k)
                        y1ii += temp2

                local_sums[0][iloc] = y10
                local_sums[1][iloc] = y1i
                local_sums[2][iloc] = y1ii
            else:
                y00 = y10
                y0i = y1i
                y0ii = y1ii

                y10 = y00 + series[i+scale] - series[i]
                
                temp1 = series[i]*<double>(k+1)
                temp2 = series[i+scale]*<double>k
                
                y1i = y0i - y00 + temp1 + temp2
                
                temp1 = temp1*<double>(k+1)
                temp2 = temp2*<double>k
                y1ii = y0ii - 2*y0i + y00 - temp1 + temp2

            a0 = c[0]*y10 + c[1]*y1ii
            temp1 = (series[i + ic + 1] - a0)
            f2 += temp1*temp1

        f2 = f2/<double>(n)

    elif(order == 4):
        iloc = 0
        if(index != 0):
            kb = <long> ((scales[index-1]-1)/2)
        else:
            kb = 1
            
        nn1 = <double>(kb-k)
        nn2 = nn1*nn1
        nn3 = nn2*nn1
        nn4 = nn2*nn2
        
        if(index == 0):
            i0 = 0
            y10 = 0; y1i = 0; y1ii = 0; y1iii = 0; y1iv = 0
        else:
            i0 = scales[index-1]
            y10 = local_sums[0][iloc]
            y1i = local_sums[1][iloc] + local_sums[0][iloc]*nn1
            y1ii = local_sums[2][iloc] + 2*local_sums[1][iloc]*nn1 + local_sums[0][iloc]*nn2
            y1iii = local_sums[3][iloc] + 3*local_sums[2][iloc]*nn1 + 3*local_sums[1][iloc]*nn2 + local_sums[0][iloc]*nn3
            y1iv = local_sums[4][iloc] + 4*local_sums[3][iloc]*nn1 + 6*local_sums[2][iloc]*nn2 + 4*local_sums[1][iloc]*nn3 + local_sums[0][iloc]*nn4
            
        for i in range(i0, scale):
            y10 += series[i]
            temp1 = series[i]*<double>(i-k)
            y1i += temp1
            temp2 = temp1*<double>(i-k)
            y1ii += temp2
            temp1 = temp2*<double>(i-k)
            y1iii += temp1
            temp2 = temp1*<double>(i-k)
            y1iv += temp2
        
        local_sums[0][iloc] = y10
        local_sums[1][iloc] = y1i
        local_sums[2][iloc] = y1ii
        local_sums[3][iloc] = y1iii
        local_sums[4][iloc] = y1iv
        
        cmat(<double> k, order, c)
        
        a0 = c[0]*y10 + c[1]*y1ii + c[2]*y1iv
        temp1 = (series[ic] - a0)
        f2 = temp1*temp1

        itemp = n - scale

        for i in range(0, itemp):
            if((i+1) % i_refresh == 0):
                iloc += 1
                if(index == 0):
                    i0 = 0
                    y10 = 0; y1i = 0; y1ii = 0; y1iii = 0; y1iv = 0
                else:
                    i0 = scales[index-1]
                    y10 = local_sums[0][iloc]
                    y1i = local_sums[1][iloc] + local_sums[0][iloc]*nn1
                    y1ii = local_sums[2][iloc] + 2*local_sums[1][iloc]*nn1 + local_sums[0][iloc]*nn2
                    y1iii = local_sums[3][iloc] + 3*local_sums[2][iloc]*nn1 + 3*local_sums[1][iloc]*nn2 + local_sums[0][iloc]*nn3
                    y1iv = local_sums[4][iloc] + 4*local_sums[3][iloc]*nn1 + 6*local_sums[2][iloc]*nn2 + 4*local_sums[1][iloc]*nn3 + local_sums[0][iloc]*nn4
                
                for j in range(i0, scale):
                    if(i + 1 + j < n):
                        y10 += series[i+1+j]
                        temp1 = series[i+1+j]*<double>(j-k)
                        y1i += temp1
                        temp2 = temp1*<double>(j-k)
                        y1ii += temp2
                        temp1 = temp2*<double>(j-k)
                        y1iii += temp1
                        temp2 = temp1*<double>(j-k)
                        y1iv += temp2

                local_sums[0][iloc] = y10
                local_sums[1][iloc] = y1i
                local_sums[2][iloc] = y1ii
                local_sums[3][iloc] = y1iii
                local_sums[4][iloc] = y1iv
            else:
                y00 = y10
                y0i = y1i
                y0ii = y1ii
                y0iii = y1iii
                y0iv = y1iv

                y10 = y00 + series[i+scale] - series[i]
                
                temp1 = series[i]*<double>(k+1)
                temp2 = series[i+scale]*<double>k
                
                y1i = y0i - y00 + temp1 + temp2
                
                temp1 = temp1*<double>(k+1)
                temp2 = temp2*<double>k
                y1ii = y0ii - 2*y0i + y00 - temp1 + temp2
                
                temp1 = temp1*<double>(k+1)
                temp2 = temp2*<double>k
                y1iii = y0iii - 3*y0ii + 3*y0i - y00 + temp1 + temp2
                
                y1iv = y0iv - 4*y0iii + 6* y0ii - 4 * y0i + y00 - temp1 *<double>(k+1) + temp2*<double>k
            
            a0 = c[0]*y10 + c[1]*y1ii + c[2]*y1iv
            temp1 = (series[i + ic + 1] - a0)
            f2 += temp1*temp1

        f2 = f2/<double>(itemp+1)

    else:
        f2 = 0
        
    return f2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double estimate_f2_q(
    double[:] series, 
    long[:] scales, 
    long index, 
    int order, 
    long i_refresh,
    double[:,:] local_sums,
    double q
    ) nogil:
    cdef double c[3]
    cdef long n, i, i0, ic, j, itemp, kb, iloc, k
    cdef double f2, a0, y00, y0i, y10, y1i, y0ii, y1ii, y0iii, y0iv
    cdef double temp, temp1, temp2, nn1, nn2, nn3, nn4, y1iii, y1iv
    cdef long scale = (scales[index])
    
    n = series.shape[0]
    f2 = 0
    k = <long> ((scale-1)/2)
    ic = k

    if(order == 0):
        y10=0
        for i in range(0, scale):
             y10 += series[i]

        cmat(<double> k, order, c)
        a0 = c[0]*y10
        temp = (series[ic] - a0)
        f2 = pow(temp*temp, q/2.0)
        itemp = n - scale

        for i in range(0, itemp):
            if((i+1) % i_refresh == 0):
                y10=0
                for j in range(1, scale+1):
                    if(i+j < n):
                        y10 += series[i+j]
            else:
                y10 += series[i+scale] - series[i]

            a0 = c[0]*y10
            temp = (series[i + ic + 1] - a0)
            f2 += pow(temp*temp, q/2.0)

        f2 = pow(f2/<double>(n), 1.0/q)
    
    return f2
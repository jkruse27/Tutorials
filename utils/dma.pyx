import cython
@cython.boundscheck(False)
@cython.wraparound(False)
import numpy as np
cimport numpy as cnp

cnp.import_array()
DTYPE = np.double
ctypedef cnp.double_t DTYPE_t


def create_scales(
        min_box: int,
        max_box: int,
        ratio: int = np.exp2(0.125)
    ) -> np.array:
    """Function that creates a list of scales in the
    specified range.

    Parameters
    ----------
    min_box: int
        Minimum scale size.
    max_box: int
        Maximum scale size.
    ratio: int, optional
        Ratio between scales. Default: 2^(1/8)

    Returns
    -------
        np.array
            Numpy array with the calculated scales.
    """
    # Define maximum value based on the parameters
    rslen = int(np.log10(max_box / min_box) / np.log10(ratio)) + 1
    # Generate list of scales by multiplying the initial value by successive
    # exponents of the ratio
    rs = np.empty((rslen,))
    rs[0] = min_box
    rs[1:] = ratio
    rs = np.cumprod(rs) + 0.5

    # Select only the values in the defined range
    rs = rs[rs < max_box]

    return ((rs//2)*2+1).astype(np.int64)

#
cpdef compute_dma(
    double[:] series,
    double[:] scales,
    int order
    ):
    """Function that calculates the detrended moving average of a series
    and returns the array of values for each scale.

    Parameters
    ----------
    series: cnp.ndarray
        Time series that is being analyzed as a numpy array of doubles.
    scales: cnp.ndarray
        The scales that the series will be analyzed at as a numpy array of doubles
    order: int
        Order of the detrending polynomial.

    Returns
    -------
        int
    """
    cdef long n, i, n_scales, i_refresh
    cdef double[:] f2 = np.zeros_like(series)

    n = series.shape[0]
    n_scales = scales.shape[0]

    if(order == 0):
        i_refresh == 50000
    elif(order == 2):
        i_refresh == 5000
    else:
        i_refresh == 100
    if(i_refresh > n): i_refresh = n

    for i in range(0, n_scales):
        f2[i] = estimate_f2(series, scales[i], order, i_refresh)

    return np.asarray(f2)

#
def cmat(k: float, order: int):
    if(order == 0):
        return 1/(2*k+1)
    elif(order == 2):
        den = 8*(k**3)+12*(k**2)-2*k-3
        c1 = (9*(k**2)+9*k-3)/den
        c2 = -15/den
        return c1, c2
    elif(order == 4):
        den = 180+72*k-800*(k**2)-320*(k**3)+320*(k**4)+128*(k**5)
        c1 = (180-750*k-525*(k**2)+450*(k**3)+225*(k**4))/den;
        c2 = (1575-1050*k-1050*(k**2))/den;
        c3 = 945/ den;
        return c1, c2, c3
    else:
        raise Exception("Invalid order. Currently only 0, 2 and 4 orders are implemented.") 


def savitzky_golay(m: float, n: float, order: int):
    if(order == 0)
        return 1/(double)((n-1)/2 + m + 1)
    elif(order == 2):

    elif(order == 4):

    else:
        raise Exception("Invalid order. Currently only 0, 2 and 4 orders are implemented.") 


cpdef estimate_f2(long[:] series, long scale, int order, long i_refresh):

void sg2(double *c0, long m, long n)
{
	double d;
	double mn,n2,n3,m2,m3,k2;
	long k;
	
	mn = 2*(double)m+(double)n;
	n2 = n*n;
	n3 = n2*n;
	m2 = m*m;
	m3 = m2*m;
	
	d = (mn+3)*(mn-3)*(mn-1)*(mn+1)*(mn+5);
	
	mn = m*n;
	
	for(k=-m;k<=(n-1)/2;k++){
		k2 = k*k;
		c0[k] = (90 + 720*k2 - 240*m + 576*k*m + 960*k2*m + 48*m2 + 1728*k*m2 + 
			960*k2*m2 + 576*m3 + 1152*k*m3 + 288*m2*m2 - 120*n - 432*k*n - 
			960*k2*n - 528*mn - 2304*k*mn - 1920*k2*mn - 864*m2*n - 
			2304*k*m2*n - 576*m3*n + 84*n2 + 576*k*n2 + 240*k2*n2 + 
			720*m*n2 + 1152*k*m*n2 + 720*m2*n2 - 72*n3 - 144*k*n3 - 
			144*m*n3 + 18*n2*n2)/d;
	}
}


double est_f2(long n, long scale){
	double f2,a0,a0L,a0R;
	double y00,y0i,y10,y1i;
	double temp,temp1,temp2,y0L,y0R;
	long i,j; /* n: data length */
	long itemp;
	long k,ic;
	
	f2 = 0;
	k = (scale-1)/2;
	ic = k+1;
	f2 = 0;

	if(ends==1){
	  /* both ends */
	  y0L=0;
	  y0R=0;
	  for(i=1;i<=k;i++){
	    y0L += y[i];
	    y0R += y[n-i+1];
	  }

	  for(i=1;i<=k;i++){
	    y0L += y[k+i];
	    y0R += y[n-i+1-k];

	    sg0(i-1 ,scale);
	    a0L = c1*y0L;
	    a0R = c1*y0R;
/******************************/
	  if(nr == 1){
		data[i] = a0L;
		data[n-i+1] = a0R;
	  }
/******************************/
	    temp1 = (y[i] - a0L);
	    temp2 = (y[n-i+1] - a0R);
	    f2 += temp1*temp1+temp2*temp2;
	  }
	}

	/* initial values */
	y10=0;
	for(i=1;i<=scale;i++){
		y10 += y[i];
	}

	cmat((double)k);

	a0 = c1*y10;
	temp = (y[ic] - a0);

/******************************/
	  if(nr == 1){
		data[ic] = a0;
	  }
/******************************/

	f2 = temp*temp;

	itemp = n-scale;
	
	for(i=1;i<=itemp;i++){
		if(i % i_refresh == 0){
			y10=0;
			for(j=1;j<=scale;j++){
				y10 += y[i+j];
			}
		}else{
			y00 = y10;
			y10 = y00 + y[i+scale]-y[i];
		}
		
		a0 = c1*y10;
/******************************/
	if(nr == 1){
		data[i+ic] = a0;
	}
/******************************/

		temp = (y[i+ic] - a0);
		f2 += temp*temp;
	}
	if(ends==1){
	  return f2/(double)(itemp+1);
	}else{
	  return f2/(double)(n);
	}
}


loc_sum0 = vector(1,(long)(n/i_refresh)+1);
loc_sum1 = vector(1,(long)(n/i_refresh)+1);
loc_sum2 = vector(1,(long)(n/i_refresh)+1);

loc_sum0 = vector(1,(long)(n/i_refresh)+1);
loc_sum1 = vector(1,(long)(n/i_refresh)+1);
loc_sum2 = vector(1,(long)(n/i_refresh)+1);
loc_sum3 = vector(1,(long)(n/i_refresh)+1);
loc_sum4 = vector(1,(long)(n/i_refresh)+1);

double est_f2(long n, long k2){
	double f2,a0,a0L,a0R;
	double y00,y0i,y0ii,y10,y1i,y1ii;
	double temp1,temp2;
	double nn1,nn2;
	double *c0_;
	long i,j,i0; /* n: data length */
	long itemp;
	long k,kb,ic,scale,iloc;

	scale = rs[k2];
	k = (scale-1)/2;
	ic = k+1;
	f2 = 0;

	if(ends==1){
	  /* both ends */
	  c0_ = vector(-k,k);

	  for(i=1;i<=k;i++){
		sg2(c0_, i-1 ,scale);
		a0L = 0;
		a0R = 0;
		for(j=1;j<=i+k;j++){
			a0L += c0_[j-i]*y[j];
			a0R += c0_[j-i]*y[n-j+1];
		}
/******************************/
	  if(nr == 1){
		data[i] = a0L;
		data[n-i+1] = a0R;
	  }
/******************************/

		temp1 = (y[i] - a0L);
		temp2 = (y[n-i+1] - a0R);
		f2 += temp1*temp1+temp2*temp2;
	  }
	free_vector(c0_,-k,k);
	}
	/* center part*/	
	/* initial values */

	iloc=1;
	kb = (rs[k2-1]-1)/2;
	nn1=(double)(kb-k);
	nn2=nn1*nn1;
	
	if(k2 == 1){
		i0 = 1;
		y10=0;
		y1i=0;
		y1ii=0;
	}else{
		i0 = rs[k2-1]+1;
		y10=loc_sum0[iloc];
		y1i=loc_sum1[iloc]+loc_sum0[iloc]*nn1;
		y1ii=loc_sum2[iloc]+2*loc_sum1[iloc]*nn1+loc_sum0[iloc]*nn2;
	}

	for(i=i0;i<=scale;i++){
		y10 += y[i];
		temp1 = y[i]*(double)(i-k-1);
		y1i += temp1;
		temp2 = temp1*(double)(i-k-1);
		y1ii += temp2;
	}
	
	loc_sum0[iloc]=y10;
	loc_sum1[iloc]=y1i;
	loc_sum2[iloc]=y1ii;

	cmat((double)k);
	a0 = c1*y10+c2*y1ii;
/******************************/
	  if(nr == 1){
		data[ic] = a0;
	  }
/******************************/


	temp1 = (y[ic] - a0);
	f2 += temp1*temp1;

	itemp = n-scale;

	for(i=1;i<=itemp;i++){
		if(i % i_refresh == 0){
			iloc++;
			if(k2 == 1){
				i0 = 1;
				y10=0;
				y1i=0;
				y1ii=0;
			}else{
				i0 = rs[k2-1]+1;
				y10=loc_sum0[iloc];
				y1i=loc_sum1[iloc]+loc_sum0[iloc]*nn1;
				y1ii=loc_sum2[iloc]+2*loc_sum1[iloc]*nn1+loc_sum0[iloc]*nn2;
			}
			
			for(j=i0;j<=scale;j++){
				y10 += y[i+j];
				temp1 = y[i+j]*(double)(j-k-1);
				y1i += temp1;
				temp2 = temp1*(double)(j-k-1);
				y1ii += temp2;
			}
			loc_sum0[iloc]=y10;
			loc_sum1[iloc]=y1i;
			loc_sum2[iloc]=y1ii;
		}else{
			y00 = y10;
			y0i = y1i;
			y0ii = y1ii;

			y10 = y00 + y[i+scale]-y[i];
			temp1 = y[i]*(double)(k+1);
			temp2 = y[i+scale]*(double)k;
			y1i = y0i - y00 + temp1 + temp2;
			y1ii = y0ii - 2*y0i + y00 - temp1 *(double)(k+1) + temp2*(double)k;
		}
		a0 = c1*y10+c2*y1ii;
/******************************/
	if(nr == 1){
		data[i+ic] = a0;
	}
/******************************/

		temp1 = (y[i+ic] - a0);
		f2 += temp1*temp1;
	}
	if(ends==1){
	  return f2/(double)(itemp+1);
	}else{
	  return f2/(double)(n);
	}
}

double est_f2(long n, long k2){
	double f2,a0;
	double y00,y0i,y0ii,y0iii,y0iv,y10,y1i,y1ii,y1iii,y1iv;
	double temp1,temp2;
	double nn1,nn2,nn3,nn4;
	long i,j,i0; /* n: data length */
	long itemp;
	long k,kb,ic,scale,iloc;
	
	scale = rs[k2];
	k = (scale-1)/2;
	ic = k+1;

	/* initial values */
	iloc=1;
	kb = (rs[k2-1]-1)/2;
	nn1=(double)(kb-k);
	nn2=nn1*nn1;
	nn3=nn2*nn1;
	nn4=nn2*nn2;
	
	if(k2 == 1){
		i0 = 1;
		y10=0;
		y1i=0;
		y1ii=0;
		y1iii=0;
		y1iv=0;
	}else{
		i0 = rs[k2-1]+1;
		y10=loc_sum0[iloc];
		y1i=loc_sum1[iloc]+loc_sum0[iloc]*nn1;
		y1ii=loc_sum2[iloc]+2*loc_sum1[iloc]*nn1+loc_sum0[iloc]*nn2;
		y1iii=loc_sum3[iloc]+3*loc_sum2[iloc]*nn1+3*loc_sum1[iloc]*nn2+loc_sum0[iloc]*nn3;
		y1iv=loc_sum4[iloc]+4*loc_sum3[iloc]*nn1+6*loc_sum2[iloc]*nn2+4*loc_sum1[iloc]*nn3+loc_sum0[iloc]*nn4;
	}

	for(i=i0;i<=scale;i++){
		y10 += y[i];
		temp1 = y[i]*(double)(i-k-1);
		y1i += temp1;
		temp2 = temp1*(double)(i-k-1);
		y1ii += temp2;
		temp1 = temp2*(double)(i-k-1);
		y1iii += temp1;
		temp2 = temp1*(double)(i-k-1);
		y1iv += temp2;
	}
	
	loc_sum0[iloc]=y10;
	loc_sum1[iloc]=y1i;
	loc_sum2[iloc]=y1ii;
	loc_sum3[iloc]=y1iii;
	loc_sum4[iloc]=y1iv;
	
	
	cmat((double)k);
	
	a0 = c1*y10+c2*y1ii+c3*y1iv;
	temp1 = (y[ic] - a0);
	f2 = temp1*temp1;

	itemp = n-scale;

	for(i=1;i<=itemp;i++){
		if(i % i_refresh == 0){
			iloc++;
			if(k2 == 1){
				i0 = 1;
				y10=0;
				y1i=0;
				y1ii=0;
				y1iii=0;
				y1iv=0;
			}else{
				i0 = rs[k2-1]+1;
				y10=loc_sum0[iloc];
				y1i=loc_sum1[iloc]+loc_sum0[iloc]*nn1;
				y1ii=loc_sum2[iloc]+2*loc_sum1[iloc]*nn1+loc_sum0[iloc]*nn2;
				y1iii=loc_sum3[iloc]+3*loc_sum2[iloc]*nn1+3*loc_sum1[iloc]*nn2+loc_sum0[iloc]*nn3;
				y1iv=loc_sum4[iloc]+4*loc_sum3[iloc]*nn1+6*loc_sum2[iloc]*nn2+4*loc_sum1[iloc]*nn3+loc_sum0[iloc]*nn4;
			}
			
			for(j=i0;j<=scale;j++){
				y10 += y[i+j];
				temp1 = y[i+j]*(double)(j-k-1);
				y1i += temp1;
				temp2 = temp1*(double)(j-k-1);
				y1ii += temp2;
				temp1 = temp2*(double)(j-k-1);
				y1iii += temp1;
				temp2 = temp1*(double)(j-k-1);
				y1iv += temp2;
			}
			loc_sum0[iloc]=y10;
			loc_sum1[iloc]=y1i;
			loc_sum2[iloc]=y1ii;
			loc_sum3[iloc]=y1iii;
			loc_sum4[iloc]=y1iv;
		}else{
			y00 = y10;
			y0i = y1i;
			y0ii = y1ii;
			y0iii = y1iii;
			y0iv = y1iv;

			y10 = y00 + y[i+scale]-y[i];
			temp1 = y[i]*(double)(k+1);
			temp2 = y[i+scale]*(double)k;
			y1i = y0i - y00 + temp1 + temp2;
			temp1 = temp1*(double)(k+1);
			temp2 = temp2*(double)k;
			y1ii = y0ii - 2*y0i + y00 - temp1 + temp2;
			temp1 = temp1*(double)(k+1);
			temp2 = temp2*(double)k;
			y1iii = y0iii - 3*y0ii + 3*y0i - y00 + temp1 + temp2;
			y1iv = y0iv - 4*y0iii + 6* y0ii - 4 * y0i + y00 - temp1 *(double)(k+1) + temp2*(double)k;
		}
		
		a0 = c1*y10+c2*y1ii+c3*y1iv;

		temp1 = (y[i+ic] - a0);
		f2 += temp1*temp1;
	}

	return f2/(double)(itemp+1);
}
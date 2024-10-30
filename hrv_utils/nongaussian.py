import numpy as np
from scipy.special import gamma
from scipy.signal import savgol_filter


def renormalize(
        x: np.array,
        std: float,
        mu: float) -> np.array:
    """Function that transforms a signal x with a certain 
    distribution so that is has standard deviation std and mean mu.

    Parameters
    ----------
    x : np.array
        Original signal to be transformed
    std : float
        New standard deviation
    mu : float
        New mean

    Returns
    -------
    np.array
        Transformed signal with standard deviation std and mean mu
    """
    return (x - np.mean(x)) / np.std(x) * std - mu


def generate_nongaussian(
        m: int,
        lmd2: float,
        n0: int
) -> np.array:
    """Function that generates a series with nongaussianity
    given by the provided variance.

    Parameters
    ----------
    m : int
        Number of levels of cascading. The final series will have
        length 2^m
    lmd2 : float
        Nongaussian index
    n0 : int
        Initial number of samples from which m layers will be applied.
        The final length of the series will be n0*2^m

    Returns
    -------
    new_series : np.array
        Nongaussian signal with length n0*2^m
    """
    lmd2e = lmd2 / m
    lmd0 = np.sqrt(lmd2e)

    x = np.exp(np.sum(
            np.array(
                [
                    np.repeat(
                        renormalize(
                            np.random.normal(size=n0*2**i),
                            lmd0,
                            lmd2e
                        ),
                        2**(m-i)
                    )
                    for i in range(m)
                ]),
            axis=0
    ))

    return np.random.normal(size=len(x)) * x


def nongaussian_index(x: np.array, q: float) -> float:
    """Function that computes the nongaussian index for a series

    Parameters
    ----------
    x : np.array
        Series that is being analyzed.
    q : float
        Moment used when evaluating the variance

    Returns
    -------
    nongaussian parameter : float
        Nongaussian parameter
    """
    k = 2/(q*(q-2))
    a = np.log(np.sqrt(np.pi)*np.mean(np.abs(x)**q)/(2**(q/2)))
    b = np.log(gamma((q+1)/2))

    return k*(a-b)


def log_amplitudee_fluctuation(x: np.array) -> float:
    """Function that computes the log amplitude fluctuation for a series

    Parameters
    ----------
    x : np.array
        Series that is being analyzed.

    Returns
    -------
    nongaussian parameter : float
        Nongaussian parameter
    """
    log_abs = np.log(np.abs(x))
    return np.mean((log_abs-np.mean(log_abs))**2)-np.pi**2/8


def nongaussian_analysis(
    series: np.array,
    scales: np.array,
    q: float = 0.25,
    m: int = 2
) -> np.array:
    """Function that computes the nongaussian index for a series for
    a set of scales.

    Parameters
    ----------
    series : np.array
        Series that is being analyzed.
    scales : np.array
        Scales being analyzed. All values must be odd numbers.
    q : float, optional
        Moment used when evaluating the variance. Default: 0.25
    m : int, optional
        Order of the savitzky-golay polynomial. Default: 2

    Returns
    -------
    nongaussian parameter : float
        Nongaussian parameter
    """
    y = np.cumsum(series - np.mean(series))

    nongaussianity = []
    curves = []

    for s in scales:
        e = y - savgol_filter(y, s, m)
        dy = e[s:]-e[:-s]

        dy = dy/np.std(dy)

        nongaussianity.append(nongaussian_index(dy, q))
        curves.append(dy)

    return nongaussianity, curves


# This code was taken from https://github.com/mlcs/iaaft
# Please refer to the original.
def surrogates(
        x, ns, tol_pc=5., verbose=True, maxiter=1E6, sorttype="quicksort"
):
    """
    Returns iAAFT surrogates of given time series.

    Parameter
    ---------
    x : numpy.ndarray, with shape (N,)
        Input time series for which IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    verbose : bool
        Show progress bar (default = `True`).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.
    sorttype : string
        Type of sorting algorithm to be used when the amplitudes of the newly
        generated surrogate are to be adjusted to the original data. This
        argument is passed on to `numpy.argsort`. Options include: 'quicksort',
        'mergesort', 'heapsort', 'stable'. See `numpy.argsort` for further
        information. Note that although quick sort can be a bit faster than 
        merge sort or heap sort, it can, depending on the data, have worse case
        spends that are much slower.

    Returns
    -------
    xs : numpy.ndarray, with shape (ns, N)
        Array containing the IAAFT surrogates of `x` such that each row of `xs`
        is an individual surrogate time series.

    See Also
    --------
    numpy.argsort

    """
    # as per the steps given in Lancaster et al., Phys. Rep (2018)
    nx = x.shape[0]
    xs = np.zeros((ns, nx))
    ii = np.arange(nx)

    # get the fft of the original array
    x_amp = np.abs(np.fft.fft(x))
    x_srt = np.sort(x)
    r_orig = np.argsort(x)

    for k in range(ns):
        # 1) Generate random shuffle of the data
        count = 0
        r_prev = np.random.permutation(ii)
        r_curr = r_orig
        z_n = x[r_prev]
        percent_unequal = 100.

        # core iterative loop
        while (percent_unequal > tol_pc) and (count < maxiter):
            r_prev = r_curr

            # 2) FFT current iteration yk, and then invert it but while
            # replacing the amplitudes with the original amplitudes but
            # keeping the angles from the FFT-ed version of the random
            y_prev = z_n
            fft_prev = np.fft.fft(y_prev)
            phi_prev = np.angle(fft_prev)
            e_i_phi = np.exp(phi_prev * 1j)
            z_n = np.fft.ifft(x_amp * e_i_phi)

            # 3) rescale zk to the original distribution of x
            r_curr = np.argsort(z_n, kind=sorttype)
            z_n[r_curr] = x_srt.copy()
            percent_unequal = ((r_curr != r_prev).sum() * 100.) / nx

            # 4) repeat until number of unequal entries between r_curr and 
            # r_prev is less than tol_pc percent
            count += 1

        if count >= (maxiter - 1):
            print("maximum number of iterations reached!")

        xs[k] = np.real(z_n)

    return xs

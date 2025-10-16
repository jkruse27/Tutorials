import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy import interpolate


def detrending(
    series: np.array,
    s: int,
    order: int = 2
) -> np.array:
    """Function that performs savitsky-golay filter detrending in a time
    series. We start by integrating the series, then for each non-overlapping
    interval of length `s`, a polynomial of order `order` is fit to that
    interval and subtracted from the raw series. The returned series is then
    the original series with each interval detrended by the savitsky golay
    approximation.

    Parameters
    ----------
    series : np.array
        Raw time series as a numpy array.
    s : int
        Length of the splits for detrending. Should be an odd number.
    order : int, optional
        Order of the polynomial to be fit. Should be an even number. Default: 2

    Returns
    -------
    new_series : np.array
        Detrended np.array with the same shape as the original data.

    Notes
    -----
    It follows the algorithm described in [1]_.

    References
    ----------
    .. [1] Kiyono, K., Struzik, Z. R., Aoyagi, N., & Yamamoto, Y. (2006).
    Multiscale probability density function analysis: non-Gaussian and
    scale-invariant fluctuations of healthy human heart rate. IEEE
    transactions on bio-medical engineering, 53(1), 95–102.
    https://doi.org/10.1109/TBME.2005.859804
    """
    return (series-savgol_filter(series, s, order))


def clean_dataset(
    df: pd.DataFrame,
    threshold_min: int = 350,
    threshold_max: int = 1500,
    max_diff: float = 0.2,
) -> pd.DataFrame:
    """
    Function that cleans the HRV data by removing outliers and adjacent points
    with too much variation in between. The removed points are replaced by
    interpolating the data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the data RRI data as the only column and the time of the
        data as the index.
    threshold_min : int, optional
        Minimum accepted value for an RRI interval. Points below this are
        discarded. Due to physical limitations, it's usually between 200-300ms.
        Default: 350.
    threshold_max : int, optional
        Maximum accepted value for an RRI interval. Points above this are
        removed. Due to physical limitations, it's usually between 1500-2000ms.
        Default: 1500.
    max_diff : float, optional
        Maximum accepted difference between adjacent RRI intervals. Points that
        have a difference over this when compared to the previous one are
        discarded. Due to physical limitations, it's usually between 200-400ms,
        or 10%-20% variation. If int, it is assumed as a difference in ms, if
        float between 0 and 1, it is assumed as a difference in percentages.
        Default: 0.2.

    Returns
    -------
    new_series : pd.DataFrame
        Clean data.
    """
    RRI = df.values
    percentage = max_diff < 1
    for ii in range(len(RRI)-1):
        diff = RRI[ii-1]-RRI[ii]

        if(percentage):
            diff /= min(RRI[ii], RRI[ii-1])

        if(diff > max_diff):
            # Ventricular premature contraction (VPC)
            if RRI[ii+1] > RRI[ii-1]:
                RRI[ii] = (RRI[ii]+RRI[ii+1])/2
                RRI[ii+1] = RRI[ii]
            # Supraventricular premature contraction (SVPC)
            else:
                RRI[ii] = (RRI[ii]+RRI[ii+1])/2

        diff = RRI[ii]-RRI[ii-1]
        if(percentage):
            diff /= min(RRI[ii], RRI[ii-1])

        if(diff > max_diff):
            # The omission of detecting R-R wave
            RRI[ii] = (RRI[ii-1] + RRI[ii+1]) / 2
        
        if(RRI[ii] < threshold_min):
            RRI[ii] = (RRI[ii-1] + RRI[ii+1]) / 2
        
        if(RRI[ii] > threshold_max):
            RRI[ii] = (RRI[ii-1] + RRI[ii+1]) / 2
    
    df.values = RRI

    return df.iloc[:, :]


def resample_hrv(
    signal: pd.DataFrame,
    fs: int,
    timestamps : np.array = None
) -> pd.DataFrame:
    """
    Function that resamples the dataset so that all points are evaluated with
    a fixed sampling rate.

    Parameters
    ----------
    signal : pd.DataFrame
        DataFrame with the data RRI data as the only column and the time of the
        data as the index.
    fs : int
        Sampling rate in Hz that the series will be interpolated by
    timestamps : np.array
        Array with the time stamps

    Returns
    -------
    new_series : pd.DataFrame
        Interpolated data.
    """
    start = signal.index[0]

    if (timestamps is None):
        timestamps = np.cumsum(signal.values, dtype=int)

    timestamps = np.insert(timestamps, 0, 0)
    RRI = np.insert(signal.values, 0, signal.values[0])
    new_time = np.arange(0, timestamps[-1], 1000/fs)
    f = interpolate.interp1d(timestamps, RRI)
    RRI_resampled = np.array(f(new_time), dtype=np.float32)

    t = [start + pd.Timedelta(x/fs, 's') for x in new_time]

    return pd.DataFrame(RRI_resampled, index=t)


def read_file(
    filename: str,
    s: int = 51,
    order: int = 4,
    low_rri: int = 300,
    high_rri: int = 1600,
    diff_rri: int = 0.2,
    detrending: bool = False,
    resampling_rate: int = 4,
    clean_data: bool = True,
    repetitions: int = 1
) -> np.array:
    """Function that read HRV data from file cleans it if so required.

    Parameters
    ----------
    filename : str or list
        Name of the csv file containing the HRV data or list with the data
    s : int, opt
        Length of the splits for detrending. Default: 41
    order : int, optional
        Order of the polynomial to be fit. Default: 4
    low_rri : int, optional
        Minimum value allowed for RRI that is used in the preprocessing.
        Default: 300
    high_rri : int, optional
        Maximum value allowed for RRI that is used in the preprocessing.
        Default: 1600
    diff_rri : int, optional
        Maximum difference between successive RRI that is used in the
        preprocessing. Default: 250
    detrending : bool, optional
        Determine whether detrending is applied or not. Default: False
    resampling_rate : int, optional
        Determines the sampling rate in Hz to use to interpolate the signal.
        If None, the signal is not interpolated. Default: None
    clean_data : bool, optional
        Whether or not to pre-process the dataset. Default: True
    repetitions : int, optional
        Number of times the cleaning process will be repeated.
    Returns
    -------
    out : np.array
        Numpy array with the time series.
    """
    if (isinstance(filename, str)):
        df = pd.read_csv(filename, header=None)
    else:
        df = pd.DataFrame(filename)

    df[0] = pd.to_numeric(df[0], 'coerce').interpolate()
    timestamps = np.cumsum(df[0])

    df['Time'] = pd.to_datetime(df[0].cumsum(), unit='ms', errors='coerce')
    df = df.set_index('Time')

    if (clean_data):
        df = clean_dataset(
            df,
            threshold_min=low_rri,
            threshold_max=high_rri,
            max_diff=diff_rri,
            repetitions=repetitions
            )

    if (resampling_rate is not None):
        df = resample_hrv(df, resampling_rate, timestamps)

    out = df.to_numpy().flatten()
    out = np.nan_to_num(out, nan=np.mean(out))

    if (detrending and (len(out) > 2*s)):
        out = detrending(
                    out,
                    s,
                    order
                )
    return out.astype(np.double)


def read_file_hourly(
    filename: str,
    initial_time: str,
    low_rri: int = 300,
    high_rri: int = 1600,
    diff_rri: int = 0.2,
    resampling_rate: int = 4,
    clean_data: bool = True,
    offset: int = None,
    freq: str = '1h',
    overlap: float = 1,
    repetitions: int = 1
) -> dict:
    """Function that read HRV data from file, splits it
     into hourly segments and cleans it if so required.

    Parameters
    ----------
    filename : str or list
        Name of the csv file containing the HRV data or list with the data
    initial_time : str
        Time to use as the origin of the recording
    low_rri : int, optional
        Minimum value allowed for RRI that is used in the preprocessing.
        Default: 300
    high_rri : int, optional
        Maximum value allowed for RRI that is used in the preprocessing.
        Default: 1600
    diff_rri : int, optional
        Maximum difference between successive RRI that is used in the
        preprocessing. Default: 250
    resampling_rate : int, optional
        Determines the sampling rate in Hz to use to interpolate the signal.
        If None, the signal is not interpolated. Default: None
    clean_data : bool, optional
        Whether or not to pre-process the dataset. Default: True
    offset : int, optional
        Amount in minutes to offset the initial position. Default: None.
    freq : str, optional
        String with the duration of each recording. Default: '1h'
    overlap : float, optional
        Float between 0 and 1 with the percentage of overlap. Default: 1
    repetitions : int, optional
        Number of times the cleaning process will be repeated.
    Returns
    -------
    out : np.array
        Numpy array with the time series.
    """
    if (isinstance(filename, str)):
        df = pd.read_csv(filename, header=None)
    else:
        df = pd.DataFrame(filename)

    df[0] = pd.to_numeric(df[0], 'coerce').interpolate()

    df['Time'] = pd.to_datetime(
        df[0].cumsum(),
        unit='ms',
        errors='coerce',
        origin=initial_time
    )

    df = df.set_index('Time')

    if (offset is not None):
        df.index = df.index + pd.to_timedelta(offset, unit='min')

    results = {}

    sep = re.split(r'(\d+)', freq)
    overlap = 1 - overlap
    dt = pd.to_timedelta(int(sep[1]), unit=sep[2])*overlap
    window = pd.to_timedelta(int(sep[1]), unit=sep[2])

    if (clean_data):
        df = clean_dataset(
            df,
            threshold_min=low_rri,
            threshold_max=high_rri,
            max_diff=diff_rri,
            repetitions=repetitions
            )

    if (resampling_rate is not None):
        df = resample_hrv(df, resampling_rate)

    step = df.index[0]-(df.index[0].minute*pd.to_timedelta(1, unit='min'))
    hour = step.hour

    while (df.index[0] >= step):
        start = np.argmax(df.index >= step)
        end = np.argmax(df.index[start:] >= step+window)+start

        out = df.iloc[start:end, :]

        if (len(out)):
            out = out.to_numpy().flatten()
            out = np.nan_to_num(out, nan=np.mean(out))

            if (hour not in results):
                results[hour] = out.astype(np.double)

        hour += overlap*int(sep[1])
        step += dt
        df = df.iloc[np.argmax(df.index >= step):, :]

    return results


def time_split(signal: np.array, freq: str) -> list:
    """Function that splits the time series into same length segments

    Parameters
    ----------
    signal : np.array
        Array with the HRV data in ms.
    freq : str
        String with the lenght of each segment.
    Returns
    -------
    out : list
        Numpy array with the time series.
    """
    df = pd.DataFrame(
            signal,
            index=pd.to_datetime(np.cumsum(signal), unit='ms')
            )

    split = df.groupby(pd.Grouper(freq=freq))

    return [
        i.to_numpy().flatten() for _, i in split
    ]

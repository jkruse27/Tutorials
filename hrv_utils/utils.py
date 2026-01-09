import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from hrv_utils import rri_utils


def parse_time_string(t_str):
    """Parses 'HH:MM:SS' string into float hours."""
    if t_str is None:
        return 0.0
    try:
        parts = t_str.split(':')
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        s = float(parts[2]) if len(parts) > 2 else 0.0
        return h + m/60.0 + s/3600.0
    except Exception:
        return 0.0


def parse_freq_to_hours(freq_str):
    """Converts frequency strings like '1h', '30min' to float hours."""
    sep = re.split(r'(\d+)', freq_str)
    val = float(sep[1])
    unit = sep[2].lower()

    if unit in ['h', 'hour', 'hours']:
        return val
    elif unit in ['m', 'min', 't', 'mins']:
        return val / 60.0
    elif unit in ['s', 'sec', 'second']:
        return val / 3600.0
    else:
        raise ValueError(f"Unknown time unit: {unit}")


def detrend_signal(
    signal: np.ndarray, s: int = 51, order: int = 4
) -> np.ndarray:
    """
    Detrends the signal using a Savitzky-Golay filter to estimate the trend.

    Parameters
    ----------
    signal : np.ndarray
        The input signal (RRI).
    s : int
        Window length for the filter (must be odd).
    order : int
        The order of the polynomial used to fit the samples.

    Returns
    -------
    np.ndarray
        The detrended signal (signal - trend).
    """
    if s % 2 == 0:
        s += 1
    if s <= order:
        raise ValueError(f"s ({s}) must be greater than poly order ({order})")

    trend = savgol_filter(signal, window_length=s, polyorder=order)

    return signal - trend


def read_file(
    filename,
    s: int = 51,
    order: int = 4,
    low_rri: int = 333,
    high_rri: int = 1600,
    diff_rri: float = 200.0,
    detrending: bool = False,
    resampling_rate: int = 2,
    clean_data: bool = True,
    return_timestamps: bool = False,
    start_hour: float = 0.0
):
    """
    Reads HRV data, cleans it using Cython optimized functions,
    optionally resamples, and optionally detrends.

    Parameters
    ----------
    filename : str or list/array
        Path to the .rri csv file (Header line 1: StartTime, Line 2+: Data)
        OR a list/array of RRI values.
    s : int, optional
        Window size for detrending filter. Default: 51
    order : int, optional
        Polynomial order for detrending. Default: 4
    low_rri : int, optional
        Minimum acceptable RRI (ms). Default: 300
    high_rri : int, optional
        Maximum acceptable RRI (ms). Default: 1600
    diff_rri : float, optional
        Maximum absolute difference allowed between successive RRIs (ms).
        Default: 200.0
    detrending : bool, optional
        If True, applies polynomial detrending. Default: False
    resampling_rate : int, optional
        Sampling rate in Hz. If None, signal is not resampled
        (returns beat-to-beat).
        Default: 4
    clean_data : bool, optional
        If True, applies artifact correction and duplicate splitting.
        Default: True
    return_timestamps : bool, optional
        If True, returns (rri, timestamps). Default: False

    Returns
    -------
    np.array or tuple
        The processed RRI signal, or (RRI, Time) if return_timestamps is True.
    """
    rri_raw = None

    if isinstance(filename, str):
        df = pd.read_csv(filename, header=None)
        rri_raw = pd.to_numeric(
            df[0], 'coerce'
            ).interpolate().values.flatten().astype(np.float64)
    else:
        if isinstance(filename, pd.DataFrame):
            rri_raw = filename.values.flatten().astype(np.float64)
        else:
            rri_raw = np.array(filename, dtype=np.float64)

    if rri_raw is None or len(rri_raw) < 2:
        raise ValueError("Input data is empty or too short.")

    time_rri = start_hour + np.cumsum(rri_raw) / 3_600_000.0

    if clean_data:
        cleaned = rri_utils.clean_rri_signal(
            rri_raw,
            time_rri,
            n_med=9,
            rri_max=float(high_rri),
            rri_min=float(low_rri),
            rri_diff=float(diff_rri)
        )
        rri_current = cleaned['rri']
        time_current = cleaned['time']
    else:
        rri_current = rri_raw
        time_current = time_rri

    if resampling_rate is not None:
        t_sec = 1.0 / resampling_rate

        resampled = rri_utils.resample_signal(
            time_current,
            rri_current,
            t_sec=t_sec
        )
        rri_current = resampled['rri']
        time_current = resampled['time']

    if detrending:
        rri_current = detrend_signal(rri_current, s=s, order=order)

    if return_timestamps:
        return rri_current, time_current

    return rri_current


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


def read_file_hourly(
    filename,
    initial_time: str,
    low_rri: int = 350,
    high_rri: int = 1500,
    diff_rri: float = 200.0,
    resampling_rate: int = 4,
    clean_data: bool = True,
    offset: int = None,
    freq: str = '1h',
    overlap: float = 0.0
) -> dict:
    """
    Reads HRV data, cleans/resamples it globally using Cython,
    and splits it into hourly (or custom freq) segments.

    Parameters
    ----------
    filename : str or list
        Path to csv or list of data.
    initial_time : str
        Start time "HH:MM:SS".
    low_rri, high_rri, diff_rri : int/float
        Cleaning parameters.
    resampling_rate : int
        Sampling rate in Hz.
    clean_data : bool
        Apply cleaning pipeline.
    offset : int
        Offset in MINUTES to skip from start.
    freq : str
        Window size ('1h', '30min').
    overlap : float
        Overlap fraction (0.0 to <1.0).
        Note: Original code had overlap=1 implying 0 step,
        assumed 0.0 default here for safety.

    Returns
    -------
    dict
        Keys: Hour index (float), Values: Numpy array of RRI segment.
    """

    start_hour = parse_time_string(initial_time)

    if offset is not None:
        start_hour += offset / 60.0

    if isinstance(filename, str):
        df = pd.read_csv(filename, header=None)
        rri_raw = pd.to_numeric(
            df[0], 'coerce'
            ).interpolate().values.flatten().astype(np.float64)
    else:
        if isinstance(filename, pd.DataFrame):
            rri_raw = filename.values.flatten().astype(np.float64)
        else:
            rri_raw = np.array(filename, dtype=np.float64)

    time_rri = np.cumsum(rri_raw) / 3_600_000.0
    time_rri = start_hour + np.concatenate(([start_hour], time_rri[:-1]))

    if clean_data:
        cleaned = rri_utils.clean_rri_signal(
            rri_raw,
            time_rri,
            rri_min=float(low_rri),
            rri_max=float(high_rri),
            rri_diff=float(diff_rri)
        )
        curr_rri = cleaned['rri']
        curr_time = cleaned['time']
    else:
        curr_rri = rri_raw
        curr_time = time_rri

    if resampling_rate is not None:
        t_sec = 1.0 / resampling_rate
        resampled = rri_utils.resample_signal(curr_time, curr_rri, t_sec=t_sec)
        curr_rri = resampled['rri']
        curr_time = resampled['time']

    results = {}

    window_size_hr = parse_freq_to_hours(freq)

    step_size_hr = window_size_hr * (1.0 - overlap)

    if step_size_hr <= 0:
        raise ValueError("Overlap resulted in zero or negative step size.")

    current_window_start = np.floor(start_hour)

    while current_window_start + window_size_hr < curr_time[0]:
        current_window_start += step_size_hr

    max_time = curr_time[-1]

    while current_window_start < max_time:
        current_window_end = current_window_start + window_size_hr
        idx_start = np.searchsorted(
            curr_time, current_window_start, side='left'
            )
        idx_end = np.searchsorted(curr_time, current_window_end, side='left')
        segment = curr_rri[idx_start:idx_end]

        if len(segment) > 0:
            if np.isnan(segment).any():
                mean_val = np.nanmean(segment)
                segment = np.nan_to_num(segment, nan=mean_val)

            results[current_window_start] = segment.astype(np.float64)

        current_window_start += step_size_hr

    return results

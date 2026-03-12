import re
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.fft import rfft, irfft
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


def iaaft_surrogates(
    x,
    n_surrogates=10,
    max_iter=1000,
    tol=1e-4,
    patience=30,
    retry=True,
    retry_patience=30,
    seed=None,
    dtype=np.float32,
    workers=-1,
    max_retries=15
):
    rng = np.random.default_rng(seed)

    x_f64 = np.asarray(x, dtype=np.float64)
    x_sorted_f64 = np.sort(x_f64)

    x = x_f64.astype(dtype)
    x_sorted = np.sort(x)

    n = x.size
    target_amp = np.abs(rfft(x, workers=workers)).astype(dtype)
    norm_target = float(np.linalg.norm(target_amp)) + 1e-30
    n_freq = target_amp.size
    _eps = dtype(1e-7)
    _fft_kw = dict(n=n, axis=1, workers=workers)

    surrogates = np.empty((n_surrogates, n), dtype=np.float64)
    spec_errors = np.full(n_surrogates, np.inf)
    converged = np.zeros(n_surrogates, dtype=bool)
    attempts = np.ones(n_surrogates, dtype=int)

    def _save_surrogate(out_row, y_row):
        order = np.argsort(y_row)
        out_row[order] = x_sorted_f64

    def _init_batch(k):
        rand_phases = np.exp(
            1j * rng.uniform(0.0, 2 * np.pi, (k, n_freq))
        ).astype(np.complex64)
        Y_freq = irfft(target_amp * rand_phases, **_fft_kw)
        order = np.argsort(Y_freq, axis=1)
        Y = np.empty((k, n), dtype=dtype)
        Y[np.arange(k)[:, None], order] = x_sorted
        return Y

    if not retry:
        _stag_tol = tol * 0.1

        Y = _init_batch(n_surrogates)
        fft_y = rfft(Y, **_fft_kw)
        prev_error = np.full(n_surrogates, np.inf)
        stagnation = np.zeros(n_surrogates, dtype=np.int32)
        active = np.ones(n_surrogates, dtype=bool)

        for _ in range(max_iter):
            if not active.any():
                break

            act = np.where(active)[0]
            fy = fft_y[act]

            phases = fy / (np.abs(fy) + _eps)
            Z = irfft(target_amp * phases, **_fft_kw)
            order = np.argsort(Z, axis=1)
            Y[act[:, None], order] = x_sorted

            fy_new = rfft(Y[act], **_fft_kw)
            fft_y[act] = fy_new

            diff = np.abs(fy_new) - target_amp
            spec_err_act = (
                np.sqrt(np.einsum("ij,ij->i", diff, diff)) / norm_target
            ).astype(np.float64)

            improvement = np.abs(prev_error[act] - spec_err_act)
            prev_error[act] = spec_err_act

            hit_tol = spec_err_act < tol
            converged[act[hit_tol]] = True
            active[act[hit_tol]] = False

            stagnated = improvement < _stag_tol
            stagnation[act[stagnated]] += 1
            stagnation[act[~stagnated]] = 0
            timed_out = (stagnation[act] >= patience) & ~hit_tol
            active[act[timed_out]] = False

        spec_errors[:] = prev_error

        for i in range(n_surrogates):
            _save_surrogate(surrogates[i], Y[i])

        return surrogates, spec_errors, converged

    _stag_tol = tol * 0.1
    pending = list(range(n_surrogates))

    while pending:
        if (max(attempts) > max_retries):
            break

        batch_idx = np.array(pending, dtype=int)
        k = len(batch_idx)

        Y = _init_batch(k)
        fft_y = rfft(Y, **_fft_kw)
        prev_error_b = np.full(k, np.inf)
        stagnation_b = np.zeros(k, dtype=np.int32)
        active_b = np.ones(k, dtype=bool)

        for _ in range(max_iter):
            if not active_b.any():
                break

            loc = np.where(active_b)[0]
            fy = fft_y[loc]

            phases = fy / (np.abs(fy) + _eps)
            Z = irfft(target_amp * phases, **_fft_kw)
            order = np.argsort(Z, axis=1)
            Y[loc[:, None], order] = x_sorted

            fy_new = rfft(Y[loc], **_fft_kw)
            fft_y[loc] = fy_new

            diff = np.abs(fy_new) - target_amp
            spec_err_loc = (
                np.sqrt(np.einsum("ij,ij->i", diff, diff)) / norm_target
            ).astype(np.float64)

            improvement = np.abs(prev_error_b[loc] - spec_err_loc)
            prev_error_b[loc] = spec_err_loc

            hit_tol = spec_err_loc < tol
            hit_loc = loc[hit_tol]
            hit_global = batch_idx[hit_loc]
            for li, gi in zip(hit_loc, hit_global):
                _save_surrogate(surrogates[gi], Y[li])
                spec_errors[gi] = prev_error_b[li]
                converged[gi] = True
            active_b[hit_loc] = False

            stagnated = improvement < _stag_tol
            stagnation_b[loc[stagnated]] += 1
            stagnation_b[loc[~stagnated]] = 0
            timed_out = (stagnation_b[loc] >= retry_patience) & ~hit_tol
            active_b[loc[timed_out]] = False

        for li, gi in enumerate(batch_idx):
            if not converged[gi]:
                spec_errors[gi] = prev_error_b[li]

        newly_converged = {int(gi) for gi in batch_idx if converged[gi]}
        for gi in newly_converged:
            pending.remove(gi)
        for gi in pending:
            if gi in set(batch_idx.tolist()):
                attempts[gi] += 1

    return surrogates, spec_errors, converged

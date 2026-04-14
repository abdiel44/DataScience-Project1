from __future__ import annotations

import math
from typing import Dict

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch


_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
}


def _clean_signal(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def _normalize_epoch_signal(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    centered = x - mean
    if std <= 0.0:
        return centered
    return centered / std


def _bandpower(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _spectral_edge_frequency(freqs: np.ndarray, psd: np.ndarray, quantile: float) -> float:
    if freqs.size == 0 or psd.size == 0:
        return 0.0
    total = float(np.sum(psd))
    if total <= 0.0:
        return 0.0
    cumsum = np.cumsum(psd) / total
    idx = int(np.searchsorted(cumsum, quantile, side="left"))
    idx = min(max(idx, 0), len(freqs) - 1)
    return float(freqs[idx])


def _hjorth_parameters(x: np.ndarray) -> Dict[str, float]:
    activity = float(np.var(x, ddof=0))
    if x.size < 2 or activity <= 0.0:
        return {
            "activity": activity,
            "mobility": 0.0,
            "complexity": 0.0,
        }

    dx = np.diff(x)
    dx_var = float(np.var(dx, ddof=0))
    if dx_var <= 0.0:
        return {
            "activity": activity,
            "mobility": 0.0,
            "complexity": 0.0,
        }

    mobility = float(np.sqrt(dx_var / activity))
    if dx.size < 2 or mobility <= 0.0:
        return {
            "activity": activity,
            "mobility": mobility,
            "complexity": 0.0,
        }

    ddx = np.diff(dx)
    ddx_var = float(np.var(ddx, ddof=0)) if ddx.size > 0 else 0.0
    complexity = float(np.sqrt(ddx_var / dx_var) / mobility) if ddx_var > 0.0 else 0.0
    return {
        "activity": activity,
        "mobility": mobility,
        "complexity": complexity,
    }


def _downsample_for_complexity(x: np.ndarray, *, max_points: int = 120) -> np.ndarray:
    if x.size <= max_points:
        return x
    step = int(np.ceil(x.size / max_points))
    return x[::step]


def _sample_entropy(x: np.ndarray, *, m: int = 2, r_ratio: float = 0.2) -> float:
    xs = _downsample_for_complexity(x)
    n = xs.size
    if n <= m + 1:
        return 0.0
    sd = float(np.std(xs, ddof=0))
    if sd <= 0.0:
        return 0.0
    r = r_ratio * sd

    def _count_matches(order: int) -> float:
        windows = np.lib.stride_tricks.sliding_window_view(xs, order)
        if windows.shape[0] < 2:
            return 0.0
        distances = np.max(np.abs(windows[:, None, :] - windows[None, :, :]), axis=2)
        upper = np.triu_indices(distances.shape[0], k=1)
        return float(np.sum(distances[upper] <= r))

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b <= 0.0 or a <= 0.0:
        return 0.0
    return float(-np.log(a / b))


def _permutation_entropy(x: np.ndarray, *, order: int = 3, delay: int = 1) -> float:
    xs = _downsample_for_complexity(x, max_points=300)
    n = xs.size - (order - 1) * delay
    if n <= 1:
        return 0.0
    patterns: Dict[tuple[int, ...], int] = {}
    for start in range(n):
        window = xs[start : start + order * delay : delay]
        key = tuple(np.argsort(window, kind="mergesort"))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.asarray(list(patterns.values()), dtype=float)
    prob = counts / counts.sum()
    entropy = -np.sum(prob * np.log2(prob))
    max_entropy = np.log2(math.factorial(order))
    if max_entropy <= 0.0:
        return 0.0
    return float(entropy / max_entropy)


def extract_epoch_signal_features(
    signal: np.ndarray,
    sfreq: float,
    *,
    prefix: str = "eeg",
    normalize_epoch: bool = False,
) -> Dict[str, float]:
    """
    Compact, reusable epoch features for monocanal EEG.

    Output names are dataset-agnostic (`eeg_*`) so that cross-dataset experiments can
    share the same feature schema even when the underlying source channel names differ.
    """
    x = _clean_signal(signal)
    if x.size == 0:
        return {}
    if normalize_epoch:
        x = _normalize_epoch_signal(x)

    out: Dict[str, float] = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x, ddof=0)),
        f"{prefix}_var": float(np.var(x, ddof=0)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(x)))),
    }

    if x.size > 1:
        zero_cross = np.count_nonzero(np.diff(np.signbit(x)))
        out[f"{prefix}_zero_crossing_rate"] = float(zero_cross / (x.size - 1))
    else:
        out[f"{prefix}_zero_crossing_rate"] = 0.0

    nperseg = min(int(max(8, round(float(sfreq) * 4.0))), x.size)
    freqs, psd = welch(x, fs=float(sfreq), nperseg=nperseg)
    total_power = _bandpower(freqs, psd, 0.5, 30.0)

    band_powers: Dict[str, float] = {}
    for band_name, (lo, hi) in _BANDS.items():
        power = _bandpower(freqs, psd, lo, hi)
        band_powers[band_name] = power
        out[f"{prefix}_bandpower_{band_name}"] = power
        out[f"{prefix}_rel_power_{band_name}"] = float(power / total_power) if total_power > 0.0 else 0.0

    alpha_power = band_powers["alpha"]
    out[f"{prefix}_theta_alpha_ratio"] = float(band_powers["theta"] / alpha_power) if alpha_power > 0.0 else 0.0

    psd_sum = float(np.sum(psd))
    if psd_sum > 0.0:
        prob = psd / psd_sum
        prob = prob[prob > 0]
        entropy = -float(np.sum(prob * np.log2(prob)))
        out[f"{prefix}_spectral_entropy"] = float(entropy / np.log2(len(prob))) if len(prob) > 1 else 0.0
    else:
        out[f"{prefix}_spectral_entropy"] = 0.0

    out[f"{prefix}_sef50"] = _spectral_edge_frequency(freqs, psd, 0.50)
    out[f"{prefix}_sef95"] = _spectral_edge_frequency(freqs, psd, 0.95)

    hjorth = _hjorth_parameters(x)
    out[f"{prefix}_hjorth_activity"] = hjorth["activity"]
    out[f"{prefix}_hjorth_mobility"] = hjorth["mobility"]
    out[f"{prefix}_hjorth_complexity"] = hjorth["complexity"]

    out[f"{prefix}_sample_entropy"] = _sample_entropy(x)
    out[f"{prefix}_permutation_entropy"] = _permutation_entropy(x)
    out[f"{prefix}_skewness"] = float(np.nan_to_num(skew(x, bias=False), nan=0.0))
    out[f"{prefix}_kurtosis"] = float(np.nan_to_num(kurtosis(x, fisher=True, bias=False), nan=0.0))

    return out

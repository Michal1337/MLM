import numpy as np
import pandas as pd
from river.datasets import WaterFlow


def generate_trend_drift(
    n_periods: int = 1000,
    n_dims: int = 1,
    drift_point: int = 600,
    slope_pre: float = 0.02,
    slope_post: float = 1.0,
    noise_std: float = 1.0,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Generate a multidimensional time series with a linear trend before `drift_point`
    and a logarithmic trend after it.
    """
    rng = np.random.default_rng(random_seed)
    t = np.arange(n_periods)

    pre = slope_pre * t
    post = slope_pre * drift_point + slope_post * np.log1p(t - drift_point)
    base = np.where(t < drift_point, pre, post)

    base = base[:, np.newaxis]  # Make it (n_periods, 1)
    noise = rng.normal(0, noise_std, size=(n_periods, n_dims))

    return base + noise


def generate_seasonal_drift(
    n_periods: int = 1000,
    n_dims: int = 1,
    drift_point: int = 600,
    amp_pre: float = 5.0,
    amp_post: float = 15.0,
    period: float = 200.0,
    noise_std: float = 2.0,
    random_seed: int = 24,
) -> np.ndarray:
    """
    Generate a multidimensional seasonal sine wave with an amplitude drift at `drift_point`.
    """
    rng = np.random.default_rng(random_seed)
    t = np.arange(n_periods)

    amps = np.where(t < drift_point, amp_pre, amp_post)
    seasonal = amps * np.sin(2 * np.pi * t / period)
    seasonal = seasonal[:, np.newaxis]

    noise = rng.normal(0, noise_std, size=(n_periods, n_dims))

    return seasonal + noise


def generate_ar1_drift(
    n_periods: int = 1000,
    n_dims: int = 1,
    drift_point: int = 600,
    phi_pre: float = 0.5,
    phi_post: float = 0.9,
    noise_std: float = 1.0,
    y0: float = 0.0,
    random_seed: int = 7,
) -> np.ndarray:
    """
    Generate a multidimensional AR(1) process with a coefficient drift at `drift_point`.
    """
    rng = np.random.default_rng(random_seed)
    y = np.empty((n_periods, n_dims))
    y[0, :] = y0

    for t in range(1, n_periods):
        phi = phi_pre if t < drift_point else phi_post
        eps = rng.normal(0, noise_std, size=n_dims)
        y[t, :] = phi * y[t - 1, :] + eps

    return y


def generate_crypto_time_series(data_path):
    df = pd.read_csv(data_path)
    data = df["close"].values
    return data


def generate_water_flow_data():
    """
    Generate water flow data using the River library.
    """
    stream = WaterFlow()
    data = list(stream.take(10_000))
    data = [item[1] for item in data]

    return np.array(data)

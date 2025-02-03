import pandas as pd
import numpy as np

def wwma(values, n):
    """
     J. Welles Wilder's EMA
    """
    return values.ewm(alpha=1/n, adjust=False).mean()

def tr(data, high='High', low='Low', close='Close', normalized=False):
    df = data.copy()
    df['tr0'] = data[high] - data[low]
    df['tr1'] = np.abs(data[high] - data[close].shift(1))
    df['tr2'] = np.abs(data[low] - data[close].shift(1))
    tr = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    if normalized:
        return tr /df[close]
    else:
        return tr

def atr(data, high='High', low='Low', close='Close', length=7, normalized=False):

    true_range = tr(data, high, low, close, normalized=False)
    atr = wwma(true_range, length)
    if normalized:
        return atr /data[close]
    else:
        return  atr

def natr(data, high, low, close, length):
    return atr(data, high,low, close, length)/close

def hurst_fd(price_series, min_window=10, max_window=100, num_windows=20, num_samples=100):
    log_returns = np.diff(np.log(price_series))
    window_sizes = np.linspace(min_window, max_window, num_windows, dtype=int)
    R_S = []

    for w in window_sizes:
        R, S = [], []
        for _ in range(num_samples):
            start = np.random.randint(0, len(log_returns) - w)
            seq = log_returns[start:start + w]
            R.append(np.max(seq) - np.min(seq))
            S.append(np.std(seq))

        R_S.append(np.mean(R) / np.mean(S))

    log_window_sizes = np.log(window_sizes)
    log_R_S = np.log(R_S)
    coeffs = np.polyfit(log_window_sizes, log_R_S, 1)
    hurst_exponent = coeffs[0]
    fractal_dimension = 2 - hurst_exponent

    return hurst_exponent, fractal_dimension

def rolling_hurst(price_series, window, min_window=10, max_window=100, num_windows=20, num_samples=100):
    return price_series.rolling(window=window).apply(lambda x: hurst_fd(x, min_window, max_window, num_windows, num_samples)[0], raw=True)

def rolling_fractal_dimension(price_series, window, min_window=10, max_window=100, num_windows=20, num_samples=100):
    return price_series.rolling(window=window).apply(lambda x: hurst_fd(x, min_window, max_window, num_windows, num_samples)[1], raw=True)


def rvol_by_time(data, length=10):
    volume = data['Volume']
    dts = volume.index
    cum_volume = volume.groupby(dts.date, sort=False).cumsum()
    prev_mean = lambda days: (
        cum_volume
        .groupby(dts.time, sort=False)
        .rolling(days, closed='left')
        .mean()
        .reset_index(0, drop=True)  # drop the level with dts.time
    )

    return cum_volume / prev_mean(length)


def kama(price, period=10, period_fast=2, period_slow=30):
    # Efficiency Ratio
    price = price.ffill()
    change = abs(price - price.shift(period))
    volatility = (abs(price - price.shift())).rolling(period).sum()
    er = change / volatility

    # Smoothing Constant
    sc_fatest = 2 / (period_fast + 1)
    sc_slowest = 2 / (period_slow + 1)
    sc = (er * (sc_fatest - sc_slowest) + sc_slowest) ** 2

    # KAMA
    kama = np.zeros_like(price)
    kama[period - 1] = price[period - 1]

    for i in range(period, len(price)):
        kama[i] = kama[i - 1] + sc[i] * (price[i] - kama[i - 1])

    kama[kama == 0] = np.nan

    return kama
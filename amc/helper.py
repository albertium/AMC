"""
To host helper functions directly connect to pricing. Less relevant helpers should go to "utils"
"""

import numpy as np
from numba import njit
import pandas as pd
from scipy.stats import norm
from sklearn import linear_model
import matplotlib.pyplot as plt
from typing import Union, Callable


def get_european_call(f: Union[float, np.ndarray],
                      k: Union[float, np.ndarray],
                      df: Union[float, np.ndarray],
                      sig: Union[float, np.ndarray],
                      t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    C = DF * (F * N(d1) - K * N(d2))
    """
    vol_time = sig * np.sqrt(t)
    d1 = np.log(f / k) / vol_time + 0.5 * vol_time
    d2 = d1 - vol_time
    return df * (f * norm.cdf(d1) - k * norm.cdf(d2))


def get_european_put(f: float, k: Union[float, np.ndarray], df: float, sig: Union[float, np.ndarray], t: float):
    """
    C - P = DF * (F - K)
    """
    return get_european_call(f, k, df, sig, t) + df * (k - f)


def get_european_call_bs(s: float, k: float, r: float, q: float, sig: float, t: float):
    """
    Black model is more natural than Black Scholes. That's why we make BS depend on Black
    """
    vol_time = sig * np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * sig ** 2) * t) / vol_time
    d2 = d1 - vol_time
    return np.exp(-q * t) * s * norm.cdf(d1) - np.exp(-r * t) * k * norm.cdf(d2)


def get_european_put_bs(s: float, k: float, r: float, q: float, sig: float, t: float):
    return get_european_call_bs(s, k, r, q, sig, t) + np.exp(-r * t) * k - np.exp(-q * t) * s


def get_implied_vol(option_func: Callable[[float, np.ndarray, float, np.ndarray, float], np.ndarray],
                    prices: Union[float, np.ndarray],
                    f: float,
                    k: Union[float, np.ndarray],
                    df: Union[float, np.ndarray],
                    t: Union[float, np.ndarray]) -> np.ndarray:
    """
    Using binary search. Should improve. See the green book
    """
    if isinstance(k, float):
        k = np.array([k])

    lb = np.ones_like(k) * 1e-10
    ub = np.ones_like(k) * 10
    imp_vol = lb

    while np.max(ub - lb) > 1e-10:
        imp_vol = (lb + ub) / 2
        new_prices = option_func(f, k, df, imp_vol, t)
        ind = new_prices > prices
        lb = np.where(ind, lb, imp_vol)
        ub = np.where(ind, imp_vol, ub)

    return imp_vol


def get_implied_vol_call(prices: np.ndarray, f: float, k: np.ndarray, df: np.ndarray, t: np.ndarray) -> np.ndarray:
    return get_implied_vol(get_european_call, prices, f, k, df, t)


def get_implied_vol_put(prices: np.ndarray, f: float, k: Union[float, np.ndarray], df: float, t: float) -> np.ndarray:
    return get_implied_vol(get_european_put, prices, f, k, df, t)


def calc_implied_vols_from_prices(data: pd.DataFrame):
    call_data = data[data.cp_flag == 'C']
    put_data = data[data.cp_flag == 'P']

    if call_data.shape != put_data.shape:
        raise RuntimeError('Uneven call and put data')

    n_rows = call_data.shape[0]
    data = pd.merge(call_data, put_data, on=['date', 'exdate', 'strike_price'], suffixes=['_c', '_p'])
    data['strike'] = data.strike_price / 1000

    if data.shape[0] != n_rows:
        raise RuntimeError('Call and put data not matching perfectly')

    data['cp_bid'] = data.best_bid_c - data.best_bid_p
    data['cp_ask'] = data.best_offer_c - data.best_offer_p
    data['cp_mid'] = (data.cp_bid + data.cp_ask) / 2
    data['abs_cp_mid'] = data.cp_mid.abs()
    data['t_exp'] = (data.exdate - data.date).dt.days / 365.25

    all_exp = sorted(data.exdate.unique())
    forwards = []

    # estimate forward and discount factor using put call parity
    for exp in all_exp:
        selected = data[data.exdate == exp].sort_values(by='abs_cp_mid').iloc[:6]  # only use near the money data
        fitted = linear_model.LinearRegression().fit(selected[['strike']], selected.cp_mid)
        forwards.append([selected.iloc[0].exdate, -fitted.coef_[0], -fitted.intercept_ / fitted.coef_[0]])

    forwards = pd.DataFrame(forwards, columns=['exdate', 'df', 'forward'])
    data = data.merge(forwards, on=['exdate'])

    if data.shape[0] != n_rows:
        raise RuntimeError('Missing forward estimates')

    # implied bid volatility
    imp_vol_bid_c = get_implied_vol_call(data.best_bid_c, data.forward, data.strike, data.df, data.t_exp)
    imp_vol_bid_p = get_implied_vol_put(data.best_bid_p, data.forward, data.strike, data.df, data.t_exp)
    imp_vol_bid = np.where(data.strike > data.forward, imp_vol_bid_c, imp_vol_bid_p)
    imp_vol_bid[np.logical_or(data.best_bid_c < 1e-10, data.best_bid_p < 1e-10)] = np.nan
    data['imp_vol_bid'] = imp_vol_bid

    # implied ask volatility
    imp_vol_ask_c = get_implied_vol_call(data.best_offer_c, data.forward, data.strike, data.df, data.t_exp)
    imp_vol_ask_p = get_implied_vol_put(data.best_offer_p, data.forward, data.strike, data.df, data.t_exp)
    data['imp_vol_ask'] = np.where(data.strike > data.forward, imp_vol_ask_c, imp_vol_ask_p)
    return data[['date', 't_exp', 'strike', 'imp_vol_bid', 'imp_vol_ask', 'forward', 'df']]\
        .sort_values(by=['date', 't_exp', 'strike'])


def plot_implied_vols(imp_vols: pd.DataFrame, n_cols=4):
    imp_vols = imp_vols.copy()
    imp_vols['moneyness'] = np.log(imp_vols.strike / imp_vols.forward)
    maturities = imp_vols.t_exp.unique()
    n_rows = int(np.ceil(len(maturities) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    fig.tight_layout(pad=3)
    axes = axes.flatten()[: len(maturities)]

    for ax, (t_exp, group) in zip(axes, imp_vols.groupby(by=['t_exp'])):
        group.plot('moneyness', 'imp_vol_bid', s=2, ax=ax, kind='scatter', c='blue')
        group.plot('moneyness', 'imp_vol_ask', s=2, ax=ax, kind='scatter', c='orange')
        ax.set_title(f'T={t_exp:.4f}')
        ax.set_ylabel('Imp Vol')

    plt.show()

@njit
def solve_tridiagonal(cp: np.ndarray, x: np.ndarray, d: np.ndarray, mat: np.ndarray, e0: float = 0, en: float = 0) -> None:
    a, b, c = mat[0], mat[1], mat[2]
    n = len(b)

    cp[0] = c[0] / b[0]  # the first a is not used in calculation
    x[0] = (d[0] - e0) / b[0]

    # forward
    last = n - 1
    for idx in range(1, last):
        den = b[idx] - a[idx] * cp[idx - 1]
        cp[idx] = c[idx] / den  # the last c is not used in calculation
        x[idx] = (d[idx] - a[idx] * x[idx - 1]) / den

    den = b[last] - a[last] * cp[last - 1]
    x[last] = (d[last] - en - a[last] * x[last - 1]) / den

    # backward
    for idx in range(n - 2, -1, -1):
        x[idx] -= cp[idx] * x[idx + 1]

"""
To host helper functions directly connect to pricing. Less relevant helpers should go to "utils"
"""

import numpy as np
from scipy.stats import norm


def get_european_call(S: float, K: float, r: float, q: float, sig: float, T: float):
    vol_time = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / vol_time
    d2 = d1 - vol_time
    return np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)


def get_european_put(S: float, K: float, r: float, q: float, sig: float, T: float):
    return get_european_call(S, K, r, q, sig, T) + np.exp(-r * T) * K - np.exp(-q * T) * S


def get_american_call(S0, vol, T, K=40, M=50, I=4096, r=0.06):
    np.random.seed(150000)  # fix the seed for every valuation
    dt = T / M  # time interval
    df = np.exp(-r * dt)  # discount factor per time time interval
    # Simulation of Index Levels
    S = np.zeros((M + 1, I), 'd')  # stock price matrix
    S[0, :] = S0  # intial values for stock price
    for t in range(1, M + 1):
        ran = np.random.standard_normal(int(I / 2))
        ran = np.concatenate((ran, -ran))  # antithetic variates
        ran = ran - np.mean(ran)  # correct first moment
        ran = ran / np.std(ran)  # correct second moment
        S[t, :] = S[t - 1, :] * np.exp((r - vol ** 2 / 2) * dt
                                       + vol * ran * np.sqrt(dt))
    h = np.maximum(K - S, 0)  # inner values for put option
    V = np.zeros_like(h)  # value matrix
    V[-1] = h[-1]
    # Valuation by LSM
    for t in range(M - 1, 0, -1):
        rg = np.polyfit(S[t, :], V[t + 1, :] * df, 5)  # regression
        C = np.polyval(rg, S[t, :])  # evaluation of regression
        V[t, :] = np.where(h[t, :] > C, h[t, :],
                           V[t + 1, :] * df)  # exercise decision/optimization
    V0 = np.sum(V[1, :] * df) / I  # LSM estimator
    return V0

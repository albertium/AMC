"""
To host helper functions directly connect to pricing. Less relevant helpers should go to "utils"
"""

import numpy as np
from scipy.stats import norm


def get_blackscholes_call(S: float, K: float, r: float, q: float, sig: float, T: float):
    vol_time = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / vol_time
    d2 = d1 - vol_time
    return np.exp(-q * T) * S * norm.cdf(d1) - np.exp(-r * T) * K * norm.cdf(d2)



import time
import numpy as np

from .engine import PricingEngine


def get_pricing_stats(engine: PricingEngine, real: float, num_steps: int, num_paths: int, repeat: int = 1000):
    start = time.time()
    ans = np.array([engine.price(num_steps=num_steps, num_paths=num_paths) for _ in range(repeat)])
    error = ans - real
    print(f'Error: {np.sqrt(np.mean(error ** 2)) / real:.3%}')
    print(f'Time used: {time.time() - start:.1f}s')

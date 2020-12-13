import numpy as np
from functools import lru_cache
from numba import int32, float32    # import the types


class PoissonCalculator:
    def __init__(self, lam):
        self.lam = lam

    @lru_cache()
    def pmf(self, k):
        return np.power(self.lam, k) * np.exp(-self.lam) / np.math.factorial(k)

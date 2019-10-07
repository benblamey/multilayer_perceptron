import numpy as np


def logit(x):
    return 1 / (1 + np.exp(-x))


def dlogit_by_dx(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2




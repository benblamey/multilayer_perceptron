import numpy as np
import matplotlib.pyplot as plt

def logit(x):
    # assert np.all(x < 1000)
    # assert np.all(x > -1000)
    return 1 / (1 + np.exp(-x))


def dlogit_by_dx(x):
    return logit(x)*(1-logit(x))



if __name__ == '__main__':
    X = np.linspace(-10, 10)
    plt.plot(X, logit(X))
    plt.plot(X, dlogit_by_dx(X))
    plt.show()
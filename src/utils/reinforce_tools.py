import numpy as np


class Explorer:
    def __init__(self, epsilon0=1) -> None:
        self.epsilon0 = epsilon0

    def __call__(self, T, t) -> bool:
        raise NotImplementedError('This is the abstract class of Explorer!')


class ConstantExplorer(Explorer):
    def __call__(self, T, t) -> bool:
        return np.random.rand() < self.epsilon0


class LinearExplorer(Explorer):
    def exam(self, T, t):
        if t < 0.2 * T:
            return self.epsilon0
        else:
            return -self.epsilon0*(t-0.2*T)/(0.8*T) + self.epsilon0

    def __call__(self, T, t) -> bool:
        if t < 0.2 * T:
            return np.random.rand() < self.epsilon0
        else:
            return np.random.rand() + self.epsilon0*(t-0.2*T)/(0.8*T) < self.epsilon0


class ExponentialExplorer(Explorer):
    def __call__(self, T, t) -> bool:
        return np.random.rand() < self.epsilon0 * np.exp(-10*t/T)

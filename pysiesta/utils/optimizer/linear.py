import numpy as np


class LinearMixer:
    """Simple linear mixing.

    x_next = x_in + alpha * (x_out - x_in)
    """

    def __init__(self, alpha: float = 0.5):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def mix(self, x_out: np.ndarray, x_in: np.ndarray) -> np.ndarray:
        return x_in + self.alpha * (x_out - x_in)

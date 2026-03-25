from __future__ import annotations

import numpy as np

from .linear import LinearMixer


class PulayMixer:
    """Pulay/DIIS mixer for fixed-point iterations.

    Uses residual r_i = x_out_i - x_in_i and solves a constrained
    least-squares problem to combine historical outputs.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        history_size: int = 6,
        regularization: float = 1e-10,
    ):
        if history_size < 1:
            raise ValueError(f"history_size must be >= 1, got {history_size}")
        self.history_size = history_size
        self.regularization = regularization
        self._linear = LinearMixer(alpha=alpha)
        self._residual_history: list[np.ndarray] = []
        self._output_history: list[np.ndarray] = []

    def _append_history(self, residual: np.ndarray, x_out: np.ndarray) -> None:
        self._residual_history.append(residual.copy().reshape(-1))
        self._output_history.append(x_out.copy().reshape(-1))

        if len(self._residual_history) > self.history_size:
            self._residual_history.pop(0)
            self._output_history.pop(0)

    def mix(self, x_out: np.ndarray, x_in: np.ndarray) -> np.ndarray:
        x_out_arr = np.asarray(x_out)
        x_in_arr = np.asarray(x_in)

        residual = x_out_arr - x_in_arr
        self._append_history(residual=residual, x_out=x_out_arr)

        m = len(self._residual_history)
        if m < 2:
            return self._linear.mix(x_out=x_out_arr, x_in=x_in_arr)

        residuals = np.stack(self._residual_history, axis=0)
        outputs = np.stack(self._output_history, axis=0)

        gram = residuals @ residuals.T
        if self.regularization > 0.0:
            gram = gram + self.regularization * np.eye(m)

        # Solve the Pulay constrained system:
        # [G  1][c] = [0]
        # [1^T 0][λ]   [1]
        mat = np.zeros((m + 1, m + 1), dtype=gram.dtype)
        mat[:m, :m] = gram
        mat[:m, -1] = 1.0
        mat[-1, :m] = 1.0

        rhs = np.zeros(m + 1, dtype=gram.dtype)
        rhs[-1] = 1.0

        try:
            coeff = np.linalg.solve(mat, rhs)[:m]
            mixed = coeff @ outputs
            return mixed.reshape(x_out_arr.shape)
        except np.linalg.LinAlgError:
            return self._linear.mix(x_out=x_out_arr, x_in=x_in_arr)

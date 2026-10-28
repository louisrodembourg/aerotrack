"""Constant-velocity Kalman filter over bounding boxes, written from the filter equations.

State (8): box centre x, y, aspect ratio a = w / h, height h, and their velocities, in
pixels (and aspect units) per frame. Measurement (4): x, y, a, h of a detection.

    predict:  x' = F x              P' = F P F^T + Q
    update:   S = H P' H^T + R      K = P' H^T S^-1
              x  = x' + K (z - H x') P  = P' - K S K^T

Noise scales with the box height, as in SORT/DeepSORT and ByteTrack (std 1/20 of the
height for positions, 1/160 for velocities; Wojke et al. 2017), so a far, small object
and a near, large one are trusted the same relative amount. The aspect ratio gets fixed
small noise: it changes slowly and is not proportional to size.

Batched over tracks (leading axis N) because prediction runs for every track every frame.
"""

from __future__ import annotations

import numpy as np

STD_POSITION = 1.0 / 20
STD_VELOCITY = 1.0 / 160
NDIM = 4


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    w, h = b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]
    return np.stack([b[..., 0] + w / 2, b[..., 1] + h / 2, w / h, h], axis=-1)


def xyah_to_xyxy(s: np.ndarray) -> np.ndarray:
    w, h = s[..., 2] * s[..., 3], s[..., 3]
    return np.stack([s[..., 0] - w / 2, s[..., 1] - h / 2, s[..., 0] + w / 2,
                     s[..., 1] + h / 2], axis=-1)


class KalmanBoxFilter:
    def __init__(self) -> None:
        self.F = np.eye(2 * NDIM)
        self.F[:NDIM, NDIM:] = np.eye(NDIM)  # position += velocity * (dt = 1 frame)
        self.H = np.eye(NDIM, 2 * NDIM)

    @staticmethod
    def _std(h: np.ndarray, pos: float, vel: float, a_pos: float, a_vel: float) -> np.ndarray:
        """Per-dimension standard deviations, (N, 8), scaled by each box height h (N,)."""
        one = np.ones_like(h)
        return np.stack([pos * h, pos * h, a_pos * one, pos * h,
                         vel * h, vel * h, a_vel * one, vel * h], axis=-1)

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A new track from one measurement (4,): zero velocity, wide velocity prior."""
        mean = np.r_[z, np.zeros(NDIM)]
        std = self._std(np.asarray(z[3]), 2 * STD_POSITION, 10 * STD_VELOCITY, 1e-2, 1e-5)
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One frame ahead for N tracks: mean (N, 8), cov (N, 8, 8)."""
        if not len(mean):
            return mean, cov
        q = np.square(self._std(mean[:, 3], STD_POSITION, STD_VELOCITY, 1e-2, 1e-5))
        Q = np.einsum("ni,ij->nij", q, np.eye(2 * NDIM))
        return mean @ self.F.T, self.F @ cov @ self.F.T + Q

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        """Correct one track (8,), (8, 8) with its matched measurement (4,)."""
        r = np.square(self._std(np.asarray(mean[3]), STD_POSITION, 0, 1e-1, 0)[:NDIM])
        S = self.H @ cov @ self.H.T + np.diag(r)
        # K = P H^T S^-1, solved rather than inverted (S is symmetric positive definite).
        K = np.linalg.solve(S, (cov @ self.H.T).T).T
        return mean + K @ (z - self.H @ mean), cov - K @ S @ K.T

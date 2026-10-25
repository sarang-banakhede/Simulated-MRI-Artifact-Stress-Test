from __future__ import annotations

import hashlib

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import affine_transform
from skimage.metrics import structural_similarity


class ArtifactSimulator:
    """Motion (k-space), Rician noise and polynomial bias-field artifacts with a deterministic pattern per case."""

    def __init__(self, motion_events: int, motion_times, phase_axis: int, bias_order: int, ssim_sigma: float):
        self.events = int(motion_events)
        self.times = tuple(motion_times)
        self.pe_axis = int(phase_axis)
        self.bias_order = int(bias_order)
        self.ssim_sigma = float(ssim_sigma)
        n = self.bias_order
        self.bias_terms = [(i, j, k) for i in range(n + 1) for j in range(n + 1 - i) for k in range(n + 1 - i - j)]

    @classmethod
    def from_config(cls, cfg) -> "ArtifactSimulator":
        a = cfg.art
        return cls(a.motion.events, a.motion.times, a.motion.phase_encode_axis, a.bias.polynomial_order, a.ssim_sigma)

    @staticmethod
    def case_rng(case: str, artifact: str) -> np.random.Generator:
        seed = int(hashlib.sha256(f"{case}|{artifact}".encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed)

    def pattern(self, case: str, artifact: str, shape) -> dict:
        rng = self.case_rng(case, artifact)
        if artifact == "motion":
            return {
                "rot": rng.uniform(-1, 1, (self.events, 3)),
                "trans": rng.uniform(-1, 1, (self.events, 3)),
                "times": np.array(self.times[: self.events]),
            }
        if artifact == "noise":
            return {
                "n1": rng.standard_normal(shape, dtype=np.float32),
                "n2": rng.standard_normal(shape, dtype=np.float32),
            }
        if artifact == "bias":
            return {"coef": rng.uniform(-1, 1, len(self.bias_terms))}
        raise ValueError(artifact)

    @staticmethod
    def _rotation(deg) -> np.ndarray:
        ax, ay, az = np.deg2rad(deg)
        rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        return rz @ ry @ rx

    def motion(self, x, a, shp, spacing):
        c = (np.array(x.shape) - 1) / 2.0
        images = [x]
        for e in range(self.events):
            rt = self._rotation(a * shp["rot"][e]).T
            t = a * shp["trans"][e] / np.array(spacing)
            images.append(
                affine_transform(x, rt, offset=c - rt @ (c + t), order=1, mode="constant", cval=0.0, output=np.float32)
            )
        n = x.shape[self.pe_axis]
        cuts = [0] + (shp["times"] * n).astype(int).tolist() + [n]
        k = sfft.fftshift(sfft.fftn(images[0]))
        for img, lo, hi in zip(images[1:], cuts[1:-1], cuts[2:]):
            sl = [slice(None)] * 3
            sl[self.pe_axis] = slice(lo, hi)
            k[tuple(sl)] = sfft.fftshift(sfft.fftn(img))[tuple(sl)]
        return np.abs(sfft.ifftn(sfft.ifftshift(k))).astype(np.float32)

    @staticmethod
    def noise(x, p, shp, mask):
        sigma = p / 100.0 * float(x[mask].mean())
        return np.sqrt((x + sigma * shp["n1"]) ** 2 + (sigma * shp["n2"]) ** 2).astype(np.float32)

    def bias_field(self, shape, coef, c):
        ax = [np.linspace(-1, 1, n) for n in shape]
        poly = np.zeros(shape)
        for u, (i, j, k) in zip(coef, self.bias_terms):
            poly += u * (ax[0][:, None, None] ** i) * (ax[1][None, :, None] ** j) * (ax[2][None, None, :] ** k)
        return np.exp(c * poly)

    def degrade(self, x, artifact: str, p: float, shp: dict, mask, spacing):
        if p <= 0:
            return x
        if artifact == "motion":
            return self.motion(x, p, shp, spacing)
        if artifact == "noise":
            return self.noise(x, p, shp, mask)
        if artifact == "bias":
            return (x * self.bias_field(x.shape, shp["coef"], p)).astype(np.float32)
        raise ValueError(artifact)

    def ssim(self, clean_pp, deg_pp, mask_c) -> float:
        vals = clean_pp[mask_c]
        _, smap = structural_similarity(
            clean_pp,
            deg_pp,
            data_range=float(vals.max() - vals.min()),
            gaussian_weights=True,
            sigma=self.ssim_sigma,
            use_sample_covariance=False,
            full=True,
        )
        return float(smap[mask_c].mean())

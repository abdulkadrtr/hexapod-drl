"""Runs the exported policy (policy.npz) with NumPy only."""
import numpy as np


class Policy:
    def __init__(self, path="policy.npz"):
        f = np.load(path)
        self.mean, self.std = f["obs_mean"], f["obs_std"]
        self.act_size = int(f["act_size"])
        self.layers = [(f[f"w{i}"], f[f"b{i}"]) for i in range(int(f["n_layers"]))]

    def act(self, obs):
        x = (np.asarray(obs, np.float32) - self.mean) / self.std
        with np.errstate(over="ignore"):
            for i, (w, b) in enumerate(self.layers):
                x = x @ w + b
                if i < len(self.layers) - 1:
                    x = x / (1.0 + np.exp(-x))
        return np.tanh(x[: self.act_size])

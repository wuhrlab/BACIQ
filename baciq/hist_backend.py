from __future__ import annotations
from pymc3.backends import base
import numpy as np


class Histogram(base.BaseTrace):
    def __init__(self, model=None, vars=None, test_point=None,
                 bin_width: float = 0.1, remove_first: int = 0):
        super().__init__('histogram', model, vars, test_point)
        self.bin_width = bin_width
        self.hist = {}
        self.rows = {}
        self.sample_count = {}
        self.remove_first = remove_first
        if 1 / self.bin_width != int(1 / self.bin_width):
            raise ValueError('Provided bin_width does not '
                             'evenly divide range of (0,1)')

    def setup(self, draws: int, chain: int) -> None:
        self.chain = chain
        self.sample_count[self.chain] = 0
        super().setup(draws, chain, sampler_vars=None)

    def record(self, point) -> None:
        vals = {}
        for varname, value in zip(self.varnames, self.fn(point)):
            vals[varname] = value.ravel()

        self.sample_count[self.chain] += 1
        if self.sample_count[self.chain] <= self.remove_first:
            return

        for k, v in vals.items():
            if k not in self.hist:
                # add a histogram for this value (chains not separate)
                num_bins = int(1 / self.bin_width)
                self.hist[k] = np.zeros((len(v), num_bins), dtype=int)
                self.rows[k] = np.array(range(len(v)))

            # increment position
            inds = np.floor(v/self.bin_width).astype(int)
            # if value == 1 will be out of range
            inds[v == 1] -= 1
            invalid = np.where((inds < 0) | (inds >= self.hist[k].shape[1]))
            if invalid[0].size != 0:
                raise ValueError('Histogram backend only supports values in '
                                 f'[0, 1], "{k}" had value of '
                                 f'{v[invalid[0][0]]}')
            self.hist[k][self.rows[k], inds] += 1

    def _slice(self, idx) -> Histogram:
        return self

    def __len__(self) -> int:
        obs = 0
        if self.chain in self.sample_count:
            obs = self.sample_count[self.chain] - self.remove_first
        return max(obs, 0)

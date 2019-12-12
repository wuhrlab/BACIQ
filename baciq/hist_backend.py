from pymc3.backends import base
import numpy as np


class Histogram(base.BaseTrace):
    def __init__(self, name, model=None, vars=None,
                 test_point=None, bin_width=0.1, remove_first=0):
        super().__init__(name, model, vars, test_point)
        self.bin_width = bin_width
        self.hist = {}
        self.cols = {}
        self.sample_count = {}
        self.remove_first = remove_first

    def setup(self, draws, chain):
        self.chain = chain
        super().setup(draws, chain, None)

    def record(self, point):
        vals = {}
        for varname, value in zip(self.varnames, self.fn(point)):
            vals[varname] = value.ravel()

        if self.chain not in self.sample_count:
            self.sample_count[self.chain] = 0
        self.sample_count[self.chain] += 1
        if self.sample_count[self.chain] <= self.remove_first:
            return

        for k, v in vals.items():
            if k not in self.hist:
                # add a histogram for this value (chains not separate)
                num_bins = int(np.ceil(1 / self.bin_width))
                self.hist[k] = np.zeros((len(v), num_bins), dtype=int)
                self.cols[k] = np.array(range(len(v)))

            # increment position
            self.hist[k][self.cols[k],
                         np.floor(v/self.bin_width).astype(int)] += 1

    def _slice(self, idx):
        return self

    def __len__(self):
        return self.sample_count[self.chain]

    def get_values(self, varname: str, burn=0, thin=1):
        return self.hist[varname]

    def point(self, idx):
        return {varname: 0 for varname in self.hist}

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

    def setup(self, draws, chains):
        super().setup(draws, chains, None)

    def record(self, point):
        # TODO why won't this work?
        vals = {}
        for varname, value in zip(self.varnames, self.fn(point)):
            vals[varname] = value.ravel()


        for k, v in vals.items():
            if k not in self.hist:
                # add a histogram for this value (chains not separate)
                num_bins = int(np.ceil(1 / self.bin_width))
                self.hist[k] = np.zeros((len(v), num_bins), dtype=int)
                self.cols[k] = np.array(range(len(v)))
                self.sample_count[k] = 0
            self.sample_count[k] += 1
            if self.sample_count[k] <= self.remove_first:
                return
            # increment position
            self.hist[k][self.cols[k],
                         np.floor(v/self.bin_width).astype(int)] += 1

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pymc3 as pm
import os
import sys
import click
import multiprocessing
from typing import Tuple
from baciq import hist_backend


def get_num_cores() -> int:
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        return multiprocessing.cpu_count()  # assume local


def get_proteins_and_indices(data: pd.DataFrame) -> Tuple[pd.Series, np.array]:
    proteins = data['Protein ID'].unique()
    idx = np.zeros(len(data), dtype=int)
    for i, p in enumerate(proteins):
        idx[data['Protein ID'] == p] = i

    return proteins, idx


class Base_Modeler(ABC):
    def __init__(self,
                 samples: int,
                 chains: int,
                 tuning: int,
                 channel: str):
        super().__init__()
        self.samples = samples
        self.chains = chains
        self.tuning = tuning
        self.channel = channel

    @abstractmethod
    def fit_quantiles(self,
                      data: pd.DataFrame,
                      quantiles: np.array) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_histogram(self,
                      data: pd.DataFrame,
                      bin_width: float) -> pd.DataFrame:
        pass


class PYMC_Model(Base_Modeler):
    def __init__(self, samples, chains, tuning, channel):
        super().__init__(samples, chains, tuning, channel)

    def fit_quantiles(self, data, quantiles):
        '''
        Fit the model with supplied data returning quantiles
        '''
        bin_width = 0.0002
        proteins, samples = self.mcmc_sample(data, bin_width=bin_width)

        # center of each bin
        bins = np.array(range(samples.shape[1])) * bin_width + bin_width / 2
        # scale counts by total, get cumsum
        quants = np.cumsum(samples / np.sum(samples, axis=1)[:, None], axis=1)

        result = None
        for q in quantiles:
            idx = np.argmin(np.abs(quants - q), axis=1)
            if result is None:
                result = bins[idx].reshape((len(idx), 1))
            else:
                result = np.hstack((result, bins[idx].reshape((len(idx), 1))))

        result = pd.DataFrame(result,
                              index=proteins,
                              columns=['{:.9g}'.format(q) for q in quantiles])
        result.index.name = "Protein ID"
        return result

    def fit_histogram(self, data, bin_width):
        '''
        Fit the model with supplied data returning histogram
        '''
        proteins, result = self.mcmc_sample(data, bin_width)

        result = pd.DataFrame(result,
                              index=proteins,
                              columns=[bin_width*i
                                       for i in range(result.shape[1])])
        result.index.name = "Protein ID"
        return result

    def mcmc_sample(self, data, bin_width):
        proteins, idx = get_proteins_and_indices(data)
        with pm.Model():
            τ = pm.Gamma('τ', alpha=7.5, beta=1)
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
            μ = BoundedNormal('μ', mu=0.5, sigma=1, shape=len(proteins))
            κ = pm.Exponential('κ', τ, shape=len(proteins))
            pm.BetaBinomial('y', alpha=μ[idx]*κ[idx], beta=(1.0-μ[idx])*κ[idx],
                            n=data['sum'], observed=data[self.channel])
            db = hist_backend.Histogram('hist', vars=[μ],
                                        bin_width=bin_width,
                                        remove_first=self.tuning)
            pm.sample(draws=self.samples,
                      tune=self.tuning,
                      chains=self.chains,
                      cores=get_num_cores(),
                      progressbar=sys.stdout.isatty(),
                      compute_convergence_checks=False,
                      trace=db)

            return proteins, db.hist['μ']

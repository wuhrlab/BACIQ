from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pystan import StanModel
import pymc3 as pm
import pickle
import os
import hist_backend
import multiprocessing


def get_num_cores():
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        return int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        return multiprocessing.cpu_count()  # assume local


# TODO need to handle protein names to rows better
class Base_Modeler(ABC):
    def __init__(self, samples, chains, tuning, channel):
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


class Stan_Model(Base_Modeler):
    def __init__(self, samples, chains, tuning, channel, pickle_name=None):
        super().__init__(samples, chains, tuning, channel)
        if pickle_name and os.path.exists(pickle_name):
            self.model = pickle.load(open(pickle_name, 'rb'))

        else:
            code = """data {
                 int<lower=0> N_;  //number of data points
                 int<lower=0> n_b;  //number of biologicaly unique proteins
                 // vector that maps peptides to proteins
                 int<lower=1, upper=n_b> condID[N_];
                 int<lower=0> y[N_];  // heads in respective coin trials
                 int<lower=0> n[N_];  // total in respective coin trials
            }
            parameters {
                 vector<lower=0, upper=1>[n_b] mu;
                 vector<lower=0>[n_b] kappa;
                 real<lower=0> tau;
            }
            transformed parameters {
                vector<lower=0>[n_b] alpha;
                vector<lower=0>[n_b] beta;
                vector<lower=0>[N_] alpha_big;
                vector<lower=0>[N_] beta_big;

                alpha= kappa .*  mu;
                beta= kappa - alpha;

                for(i in 1:N_){

                  alpha_big[i] <- alpha[condID[i]];
                  beta_big[i] <- beta[condID[i]];
                  }

             }
            model {
                tau ~ gamma(7.5, 1);

                mu ~ normal(0.5, 1);
                kappa ~ exponential(tau);

                y ~ beta_binomial(n, alpha_big, beta_big);
            }
            """

            self.model = StanModel(model_name='baciq', model_code=code)
            if pickle_name:
                with open(pickle_name, 'wb') as f:
                    pickle.dump(self.model, f)

    def fit_quantiles(self, data, quantiles):
        samples = self.mcmc_sample(data)
        # want sample X quantile, hence transpose
        return np.quantile(samples, quantiles, axis=0).T

    def fit_histogram(self, data, bin_width):
        samples = self.mcmc_sample(data)
        num_bins = int(np.ceil(1 / bin_width))
        # want sample X count, hence transpose
        return np.apply_along_axis(lambda r: np.histogram(r, bins=num_bins)[0],
                                   arr=samples,
                                   axis=0).T

    def mcmc_sample(self, data):
        idx = pd.Categorical(data['Protein ID']).codes
        groups = np.unique(idx)
        fit = self.model.sampling(
            chains=self.chains,
            iter=self.samples + self.tuning,
            warmup=self.tuning,
            refresh=-1,
            data={'N_': len(data),  # number of peptides
                  'n_b': len(groups),  # number of proteins
                  'condID': idx + 1,  # stan is 1-based
                  'y': data[self.channel],
                  'n': data['sum']})
        return fit.extract(pars=["mu"])["mu"]


class PYMC_Model(Base_Modeler):
    def __init__(self, samples, chains, tuning, channel):
        super().__init__(samples, chains, tuning, channel)

    def fit_quantiles(self, data, quantiles):
        '''
        Fit the model with supplied data returning quantiles
        '''
        bin_width = 0.0002
        samples = self.mcmc_sample(data, bin_width=bin_width)

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

        return result

    def fit_histogram(self, data, bin_width):
        '''
        Fit the model with supplied data returning histogram
        '''
        result = self.mcmc_sample(data, bin_width)

        return result

    def mcmc_sample(self, data, bin_width):
        idx = pd.Categorical(data['Protein ID']).codes
        groups = np.unique(idx)
        with pm.Model():
            τ = pm.Gamma('τ', alpha=7.5, beta=1)
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
            μ = BoundedNormal('μ', mu=0.5, sigma=1, shape=len(groups))
            κ = pm.Exponential('κ', τ, shape=len(groups))
            pm.BetaBinomial('y', alpha=μ[idx]*κ[idx], beta=(1.0-μ[idx])*κ[idx],
                            n=data['sum'], observed=data[self.channel])
            db = hist_backend.Histogram('hist', vars=[μ],
                                        bin_width=bin_width,
                                        remove_first=self.tuning)
            try:
                pm.sample(draws=self.samples,
                          tune=self.tuning,
                          chains=self.chains,
                          cores=get_num_cores(),
                          progressbar=True,  # TODO false when done testing, or make verbose?
                          trace=db)
            except NotImplementedError:
                pass  # since the db doesn't support slice

            return db.hist['μ']

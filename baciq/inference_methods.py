from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pystan import StanModel
import pymc3 as pm
import pickle
import os
import hist_backend


class Base_Modeler(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit_quantiles(self,
                      data: pd.DataFrame,
                      channel1: str,
                      quantiles: np.array,
                      samples: int,
                      chains: int) -> np.array:
        pass


class Stan_Single(Base_Modeler):
    def __init__(self, pickle_name):
        super().__init__()
        if os.path.exists(pickle_name):
            self.model = pickle.load(open(pickle_name, 'rb'))

        else:
            code = """data {
                 int<lower=2> J;          // number of coins
                 int<lower=0> y[J];       // heads in respective coin trials
                 int<lower=0> n[J];       // total in respective coin trials
            }
            parameters {
                 real<lower = 0, upper = 1> mu;
                 real<lower = 0> kappa;
            }
            transformed parameters {
                real<lower=0> alpha;
                real<lower=0> beta;

                alpha = kappa * mu;
                beta = kappa - alpha;
            }
            model {
                mu ~ uniform(0, 1);
                kappa ~ exponential(0.05);
                y ~ beta_binomial(n, alpha, beta);
            }

            """
            self.model = StanModel(model_name='baciq', model_code=code)
            with open(pickle_name, 'wb') as f:
                pickle.dump(self.model, f)

    def fit_quantiles(self, data, channel1, quantiles, samples, chains):
        fit = self.model.sampling(
            chains=chains, iter=samples, warmup=5000, refresh=-1,
            data={'J': len(data),  # Number of peptides
                  'y': data[channel1],
                  'n': data['sum']})
        sim = fit.extract()
        quants = pd.DataFrame(sim["mu"], columns=['mu']).quantile(quantiles)
        return quants['mu'].values


class PYMC_Single(Base_Modeler):
    def __init__(self):
        super().__init__()

    def fit_quantiles(self, data, channel1, quantiles, samples, chains):
        with pm.Model():
            μ = pm.Uniform('μ', 0, 1)
            κ = pm.Exponential('κ', 0.05)
            θ = pm.Beta('θ', alpha=μ*κ, beta=(1.0-μ)*κ)
            pm.Binomial('y', p=θ, observed=data[channel1], n=data['sum'])
            trace = pm.sample(samples, init='advi+adapt_diag', chains=chains)
            return np.quantile(trace.get_values('μ'), quantiles)


class PYMC_Multiple(Base_Modeler):
    def __init__(self):
        super().__init__()

    def fit_quantiles(self, data, channel1, quantiles,
                      samples, chains):
        '''
        Fit the model with supplied data.  If quantiles is set to a single
        number, will return a histogram of the samples, otherwise the quantiles
        will be returned.
        '''
        bin_width = 0.0002
        samples = self.mcmc_sample(data, channel1, samples,
                                   chains, bin_width=bin_width)

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

    def fit_histogram(self, data, channel1, bin_width,
                      samples, chains):
        '''
        Fit the model with supplied data.  If quantiles is set to a single
        number, will return a histogram of the samples, otherwise the quantiles
        will be returned.
        '''
        result = self.mcmc_sample(data, channel1, samples,
                                  chains, bin_width)

        return result

    def mcmc_sample(self, data, channel1, samples, chains, bin_width):
        '''
        fit the model, writing samples to sqlite database
        '''
        idx = pd.Categorical(data['Protein ID']).codes
        groups = np.unique(idx)
        with pm.Model():
            τ = pm.Gamma('τ', alpha=7.5, beta=1)
            BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
            μ = BoundedNormal('μ', mu=0.5, sigma=1, shape=len(groups))
            κ = pm.Exponential('κ', τ, shape=len(groups))
            pm.BetaBinomial('y', alpha=μ[idx]*κ[idx], beta=(1.0-μ[idx])*κ[idx],
                            n=data['sum'], observed=data[channel1])
            db = hist_backend.Histogram('hist', vars=[μ],
                                        bin_width=bin_width,
                                        remove_first=1000)
            try:
                pm.sample(samples, discard_tuned_samples=True,
                          tune=1000, chains=chains, trace=db)
            except NotImplementedError:
                pass  # since the db doesn't support slice

            return db.hist['μ']

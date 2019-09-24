from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pystan import StanModel
import pymc3 as pm
import pickle
import os


class Base_Modeler(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit_quantiles(self,
                      data: pd.DataFrame,
                      channel1: str,
                      quantiles: np.array) -> np.array:
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

    def fit_quantiles(self, data, channel1, quantiles):
        fit = self.model.sampling(
            chains=1, iter=5005000, warmup=5000, refresh=-1,
            data={'J': len(data),  # Number of peptides
                  'y': data[channel1],
                  'n': data['sum']})
        sim = fit.extract()
        quants = pd.DataFrame(sim["mu"], columns=['mu']).quantile(quantiles)
        return quants['mu'].values


class PYMC_Single(Base_Modeler):
    def __init__(self):
        super().__init__()

    def fit_quantiles(self, data, channel1, quantiles):
        with pm.Model():
            μ = pm.Uniform('μ', 0, 1)
            κ = pm.Exponential('κ', 0.05)
            θ = pm.Beta('θ', alpha=μ*κ, beta=(1.0-μ)*κ)
            pm.Binomial('y', p=θ, observed=data[channel1], n=data['sum'])
            trace = pm.sample(5000, init='advi+adapt_diag')
            return np.quantile(trace.get_values('μ'), quantiles)


class PYMC_Multiple(Base_Modeler):
    def __init__(self):
        super().__init__()

    def fit_quantiles(self, data, channel1, quantiles):
        idx = pd.Categorical(data['Protein ID']).codes
        groups = np.unique(idx)
        with pm.Model():
            τ = pm.Gamma('τ', alpha=3, beta=0.1)
            # τ = pm.Gamma('τ', alpha=5, beta=0.01)
            μ = pm.Uniform('μ', 0, 1, shape=len(groups))
            κ = pm.Exponential('κ', τ, shape=len(groups))
            # κ = pm.Exponential('κ', 0.05, shape=len(groups))
            θ = pm.Beta('θ', alpha=μ*κ, beta=(1.0-μ)*κ, shape=len(groups))
            pm.Binomial('y', p=θ[idx], observed=data[channel1], n=data['sum'])
            trace = pm.sample(50000, init='advi+adapt_diag')
            return np.quantile(trace.get_values('μ'), quantiles, axis=0)

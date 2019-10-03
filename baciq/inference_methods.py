from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pystan import StanModel
import pymc3 as pm
import pickle
import os
import sqlalchemy
import sqlite3


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
                      samples, chains, db='tempMC.sqlite'):
        '''
        Fit the model with supplied data.  If quantiles is set to a single
        number, will return a histogram of the samples, otherwise the quantiles
        will be returned.
        '''
        self.mcmc_sample(data, channel1, samples, chains, db)
        engine = sqlalchemy.create_engine(f'sqlite:///{db}')
        with engine.connect() as conn, conn.begin():
            results = pd.read_sql_table('μ', engine)

        return np.quantile(results.iloc[:, 3:],
                           quantiles, axis=0)

    def fit_histogram(self, data, channel1, bin_width,
                      samples, chains, db='tempMC.sqlite'):
        '''
        Fit the model with supplied data.  If quantiles is set to a single
        number, will return a histogram of the samples, otherwise the quantiles
        will be returned.
        '''
        self.mcmc_sample(data, channel1, samples, chains, db)
        bins = int(np.ceil(1 / bin_width))
        result = None

        engine = sqlalchemy.create_engine(f'sqlite:///{db}')
        with engine.connect() as conn, conn.begin():
            for df in pd.read_sql_table('μ', engine, chunksize=1000):
                hist = np.apply_along_axis(
                    lambda x: np.histogram(x, bins, range=(0, 1))[0],
                    axis=0,
                    arr=df.iloc[:, 3:].values)
                if result is None:
                    result = hist
                else:
                    result += hist

        return np.transpose(result)

    def mcmc_sample(self, data, channel1, samples, chains, db):
        '''
        fit the model, writing samples to sqlite database
        '''
        idx = pd.Categorical(data['Protein ID']).codes
        groups = np.unique(idx)
        with pm.Model():
            τ = pm.Gamma('τ', alpha=3, beta=0.1)
            μ = pm.Uniform('μ', 0, 1, shape=len(groups))
            κ = pm.Exponential('κ', τ, shape=len(groups))
            θ = pm.Beta('θ', alpha=μ*κ, beta=(1.0-μ)*κ, shape=len(groups))
            pm.Binomial('y', p=θ[idx], observed=data[channel1], n=data['sum'])
            db = pm.backends.SQLite(db, vars=[μ])
            try:
                pm.sample(samples, discard_tuned_samples=True,
                          tune=500, chains=chains, trace=db)
            except sqlite3.ProgrammingError:
                # because pymc3 doesn't support sqlite really...
                pass

    def batch_simulate(model, samples, chains, batch_size):
        # TODO this isn't working :(
        with model:
            trace = pm.sample(batch_size, progressbar=False,
                              discard_tuned_samples=True, chains=chains)
        yield trace
        tot = batch_size
        while tot < samples:
            with model:
                trace = pm.sample(batch_size, tune=0, init=None,
                                  progressbar=False, trace=trace)
            tot += batch_size
            yield trace

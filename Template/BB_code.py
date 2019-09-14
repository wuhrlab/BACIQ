import pandas as pd
import numpy as np
import sys
import pystan


def main(infile='SampleInput.csv', outfile='Output/outFile_hist.csv'):
    channel1 = sys.argv[1]
    channel2 = sys.argv[2]
    scaling = float(sys.argv[3])
    conf = float(sys.argv[4]) / 2
    confidence = {
        'low': 0.5 - conf,
        'high': 0.5 + conf,
    }

    # The model and its compilation
    code = """
    data {
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
    stanmodel = pystan.StanModel(model_name='baciq', model_code=code)

    output = None  # pd.DataFrame

    for name, df_temp in read_df(infile, channel1, channel2, scaling):

        fit = stanmodel.sampling(
            chains=1, seed=1, iter=5005000, warmup=5000, refresh=-1,
            data={'J': len(df_temp),  # Number of peptides
                  'y': df_temp[channel1],
                  'n': df_temp['sum']})
        sim = fit.extract()
        df_temp2 = pd.DataFrame(sim["mu"]).T
        df_temp3 = df_temp2.quantile(
            [confidence['low'], 0.5, confidence['high']], axis=1).T
        df_temp3['index'] = name

        if output:
            output = pd.concat([output, df_temp3], axis=1)
        else:
            output = df_temp3

    # this is necessary to handle floating point issues of columns
    output.columns = [
        str(confidence['low']),
        str(0.5),
        str(confidence['high']),
        'index']
    # and %g handles row entries
    output.to_csv(outfile, index=False, float_format='%.9g')


def read_df(infile, channel1, channel2, multiplier):
    baciq = pd.read_csv(infile)

    # Multiply by the factor
    baciq.iloc[:, 1:] = np.around(baciq.iloc[:, 1:] * multiplier).astype(int)
    # convert values < 1 to 1
    baciq.iloc[:, 1:] = baciq.iloc[:, 1:].applymap(lambda x: 1 if x < 1 else x)

    # Sum up the information of requested channels
    if channel2 == 'sum':
        baciq[channel2] = baciq.iloc[:, 1:].sum(axis=1)
    else:
        baciq['sum'] = baciq[channel2] + baciq[channel1]

    baciq = baciq[['Protein ID', channel1, 'sum']]

    for name, df in baciq.groupby('Protein ID'):
        if len(df) > 1:
            yield name, df


if __name__ == '__main__':
    main()

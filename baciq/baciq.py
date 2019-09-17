import pandas as pd
import numpy as np
import pystan
import click


@click.command()
@click.option('-c1', '--channel1',
              help='Column to use as "heads" in beta-binomial')
@click.option('-c2', '--channel2',
              help='Column to use as "tails" in beta-binomial or "sum"')
@click.option('-s', '--scaling', default=1.0,
              help='Value to scale input data')
@click.option('-c', '--confidence', default=0.95,
              help='Confidence level to return')
@click.option('-i', '--infile', type=click.File('r'),
              help='Input csv file containing counts data')
@click.option('-o', '--outfile', type=click.File('w'),
              help='Output file')
def main(channel1, channel2, scaling, confidence, infile, outfile):
    confidence = {
        'low': 0.5 - confidence / 2,
        'high': 0.5 + confidence / 2,
    }

    # The model and its compilation
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
    stanmodel = pystan.StanModel(model_name='baciq', model_code=code)

    # formatting is necessary to handle floating point issues of columns
    lo = f'{confidence["low"]:.9g}'
    mid = str(0.5)
    hi = f'{confidence["high"]:.9g}'

    output = {  # Dict for pd
        'Protein ID': [],
        lo: [],
        mid: [],
        hi: []
    }

    for name, df_temp in read_df(infile, channel1, channel2, scaling):
        print(name)
        fit = stanmodel.sampling(
            chains=1, seed=2, iter=5005000, warmup=5000, refresh=-1,
            data={'J': len(df_temp),  # Number of peptides
                  'y': df_temp[channel1],
                  'n': df_temp['sum']})
        sim = fit.extract()
        quants = pd.DataFrame(sim["mu"], columns=['mu']).quantile(
            [confidence['low'], 0.5, confidence['high']])

        output['Protein ID'].append(name)
        output[lo].append(quants.iloc[0, 0])
        output[mid].append(quants.iloc[1, 0])
        output[hi].append(quants.iloc[2, 0])

    # %g handles floating point issues in row entries
    pd.DataFrame(output).to_csv(
        outfile, index=False, float_format='%.9g')


def read_df(infile, channel1, channel2, multiplier):
    baciq = pd.read_csv(infile)

    # Multiply by the factor
    numeric_columns = baciq.select_dtypes(include=['number']).columns
    baciq[numeric_columns] = np.around(baciq[numeric_columns]
                                       * multiplier).astype(int)
    # convert values < 1 to 1
    baciq[numeric_columns] = baciq[numeric_columns].applymap(
        lambda x: 1 if x < 1 else x)

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

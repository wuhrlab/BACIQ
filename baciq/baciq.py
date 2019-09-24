import pandas as pd
import numpy as np
import click
import inference_methods


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

    # model = inference_methods.Stan_Single('stan_single.pkl')
    # model = inference_methods.PYMC_Single()
    # grouped = True

    model = inference_methods.PYMC_Multiple()
    grouped = False

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

    for name, df in read_df(infile, channel1, channel2, scaling, grouped):
        print(name)
        quants = model.fit_quantiles(
            df, channel1, [confidence['low'], 0.5, confidence['high']])

        if grouped:
            output['Protein ID'].append(name)
            output[lo].append(quants[0])
            output[mid].append(quants[1])
            output[hi].append(quants[2])
        else:
            output['Protein ID'] = df['Protein ID'].unique()
            output[lo] = quants[0, :]
            output[mid] = quants[1, :]
            output[hi] = quants[2, :]

    # %g handles floating point issues in row entries
    pd.DataFrame(output).to_csv(
        outfile, index=False, float_format='%.9g')


def read_df(infile, channel1, channel2, multiplier, grouped=True):
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

    if grouped:
        for name, df in baciq.groupby('Protein ID'):
            if len(df) > 1:
                yield name, df
    else:
        yield 'all', baciq


if __name__ == '__main__':
    main()

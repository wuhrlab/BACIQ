import pandas as pd
import numpy as np
import click
import inference_methods
import math


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
              help='Output file, can be csv or pkl')
@click.option('-b', '--bin-width', type=float, default=None,
              help='Bin width for histogram output.  If specified, '
              'confidence will be ignored')
@click.option('--samples', default=1000,
              help='Number of samples for MCMC per chain')
@click.option('--chains', default=5,
              help='Number of MCMC chains')
@click.option('--tuning', default=1000,
              help='Number of burn in samples per chain')
@click.option('--batch-size', default=None, type=int,
              help='Number of proteins to run at once')
def main(channel1, channel2, scaling, confidence, bin_width,
         samples, chains, tuning, batch_size,
         infile, outfile):

    confidence = {
        'low': 0.5 - confidence / 2,
        'high': 0.5 + confidence / 2,
    }

    model = inference_methods.PYMC_Model(samples, chains, tuning, channel1)

    # TODO reorganize to put read_df in another module, and most of this
    # code in inference methods
    # TODO add tests for everything!
    if bin_width is not None:
        print('Estimating histogram')
    else:
        print('Estimating quantiles')

    for i, df in enumerate(read_df(infile, channel1, channel2, scaling,
                                   batch_size=batch_size)):
        if bin_width is not None:
            output = model.fit_histogram(df, bin_width)
        else:
            output = model.fit_quantiles(
                df, [confidence['low'], 0.5, confidence['high']])

        # create first time, write on subsequent
        if i == 0:
            output.to_csv(outfile, float_format='%.9g')
        else:
            output.to_csv(outfile, float_format='%.9g',
                          mode='a', header=False)


def read_df(infile, channel1, channel2, multiplier, batch_size=None):
    baciq = pd.read_csv(infile, dtype={'Protein ID': 'category'}
                        ).dropna(axis='columns', how='all')

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

    if batch_size:
        proteins = baciq['Protein ID'].cat.categories
        if batch_size < len(proteins):
            num_batches = math.ceil(len(proteins) / batch_size)
            for i, prots in enumerate(np.array_split(proteins, num_batches)):
                print(f'Batch {i+1} of {num_batches}. {len(prots)} proteins')
                yield baciq.loc[baciq['Protein ID'].isin(prots)]

        else:
            yield baciq

    else:
        yield baciq


if __name__ == '__main__':
    main()

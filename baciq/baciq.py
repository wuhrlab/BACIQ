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

    # TODO need to properly handle protein ids and multiple iterations of
    # read df
    # TODO reorganize to put read_df in another module, and most of this
    # code in inference methods
    # TODO add tests for everything!
    if bin_width is not None:
        print('Estimating histogram')
        for df in read_df(infile, channel1, channel2, scaling,
                          batch_size=batch_size):
            hist = model.fit_histogram(df, bin_width)

        output = pd.DataFrame(
            hist, index=df['Protein ID'].unique(),
            columns=[bin_width*i for i in range(hist.shape[1])]
        )

    else:
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

        print('Estimating quantiles')
        for df in read_df(infile, channel1, channel2, scaling,
                          batch_size=batch_size):
            quants = model.fit_quantiles(
                df, [confidence['low'], 0.5, confidence['high']])

            output['Protein ID'] = df['Protein ID'].unique()
            output[lo] = quants[:, 0]
            output[mid] = quants[:, 1]
            output[hi] = quants[:, 2]

        output = pd.DataFrame(output)

    if outfile.name.endswith('.csv'):
        # %g handles floating point issues in row entries
        output.to_csv(outfile,
                      index=(bin_width is not None),
                      float_format='%.9g')
    else:
        output.to_pkl(outfile)


def read_df(infile, channel1, channel2, multiplier, batch_size=None):
    # TODO walk through this to see how and when sorting is an issue
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
    baciq = baciq.sort_values(by='Protein ID')

    # TODO make this yield all proteins (shuffle then return slices)
    if batch_size:
        counts = baciq['Protein ID'].value_counts()
        proteins = sorted(counts[counts > 1].index.tolist())
        if batch_size < len(proteins):
            np.random.seed(0)
            proteins = np.random.choice(proteins, batch_size, replace=False)
            # TODO num_batches = math.ceil(len(proteins))
            # for prots in np.array_split(proteins, num_batches):

            baciq = baciq.loc[
                baciq['Protein ID'].isin(proteins)]

            yield baciq

        else:
            yield baciq

    else:
        yield baciq


if __name__ == '__main__':
    main()

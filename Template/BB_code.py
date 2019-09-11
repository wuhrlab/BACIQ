import pandas as pd
import numpy as np
import sys
import pystan


def main():
    current_code_n = int(sys.argv[1])
    n_codes = int(sys.argv[2])
    flag = int(sys.argv[3])
    int1 = int(sys.argv[4])
    int2 = int(sys.argv[5])
    float3 = float(sys.argv[6])
    conf = float(sys.argv[7])
    loconf = float(1-conf) / 2
    highconf = 1 - loconf
    print(current_code_n, n_codes, flag, int1, int2, float3)

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
        kappa ~ exponential(0.05); // uniform(1,100);
        y ~ beta_binomial(n, alpha, beta);
    }

    """
    sm = pystan.StanModel(model_name='baciq', model_code=code)

    flag1 = 0

    ch1 = 'channel1'
    ch2 = 'channel2'

    for name, df_temp in read_df(float3, current_code_n,
                                 n_codes, flag, int1, int2):

        df_temp['sum'] = df_temp.loc[:, ch1] + df_temp.loc[:, ch2]

        data_temp = {'J': len(df_temp),  # Number of peptides
                     # Value of channel 1 for that peptide
                     'y': df_temp[ch1].values.tolist(),
                     # Value of total throws for that peptide
                     'n': df_temp['sum'].values.tolist()}
        fit = sm.sampling(data=data_temp, chains=1, seed=1,
                          iter=5005000, warmup=5000, refresh=-1)
        sim = fit.extract()
        df_temp2 = pd.DataFrame(sim["mu"]).T
        if (flag1 == 0):
            df_temp3 = df_temp2.quantile([loconf, 0.5, highconf], axis=1).T
            df_temp3['index'] = name
            final_df = df_temp3
            flag1 = 1
        else:
            df_temp3 = df_temp2.quantile([loconf, 0.5, highconf], axis=1).T
            df_temp3['index'] = name
            final_df = pd.concat([final_df, df_temp3], axis=1)

    df_export = final_df
    df_export.to_csv('Output/outFile_hist%d.csv' % (current_code_n))


def read_df(float3, current_code_n, n_codes,
            flag, int1, int2, infile='SampleInput.csv'):
    baciq = pd.read_csv(infile)

    # Multiply by the factor
    df_array = np.around(baciq.iloc[:, 1:].values * float3)
    df_array[df_array < 1] = 1

    # The columns get named starting 0 so we can genralize the code
    df_values = pd.DataFrame(df_array, dtype=int)

    # Change back the numpy format to dataframe after the manipulation
    df = pd.concat([baciq['Protein ID'], df_values], axis=1)

    df_manycoinsdata = df

    # Split the proteins into multiple files to parallelize
    df_manycoinsdata1 = df_manycoinsdata.sort_values(
        ['Protein ID'], ascending=True).reset_index().drop(['index'], axis=1)
    df_manycoinsdata1['index'] = np.array(range(len(df_manycoinsdata1)))
    df_index = df_manycoinsdata1.groupby(
        ['Protein ID']).agg({'index': [min, max]}).reset_index()
    df_index['row'] = np.array(range(len(df_index)))

    divisions = len(df_index) / n_codes
    spliced = df_index[divisions*(current_code_n-1):divisions*(current_code_n)]
    min1 = list(spliced['index']['min'])[0]
    max1 = list(spliced['index']['max'])[-1]
    if (current_code_n == n_codes):
        df_manycoinsdata = df_manycoinsdata1[min1:len(df_manycoinsdata1)]
    else:
        df_manycoinsdata = df_manycoinsdata1[min1:max1+1]

    # Sum up the information of rest of the channels if flag=0
    if flag == 0:
        temporary1 = df_manycoinsdata[['Protein ID', int1]].rename(
            columns={int1: 'channel1'})
        temporary2 = pd.DataFrame(
            df_manycoinsdata.drop(
                ['Protein ID', 'index', int1], axis=1
            ).sum(axis=1)).rename(columns={0: 'channel2'})
        df_manycoinsdata = pd.concat([temporary1, temporary2], axis=1)
    else:
        temporary1 = df_manycoinsdata[['Protein ID', int1, int2]]
        df_manycoinsdata = temporary1.rename(
            columns={int1: 'channel1', int2: 'channel2'})

    for name, df in df_manycoinsdata.groupby('Protein ID'):
        if len(df) > 1:
            yield name, df


if __name__ == '__main__':
    main()

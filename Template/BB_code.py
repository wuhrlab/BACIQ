





import pandas as pd 
import numpy as np 

import sys 

current_code_n=int(sys.argv[1])
n_codes=int(sys.argv[2])
flag=int(sys.argv[3])
int1=int(sys.argv[4])
int2=int(sys.argv[5])
float3=float(sys.argv[6]) 
conf= float(sys.argv[7])
loconf=float(1-conf)/ 2 
highconf= 1- loconf 
print current_code_n, n_codes, flag, int1, int2, float3

baciq=pd.read_csv('SampleInput.csv')


# Multiply by the factor
df_array=baciq[['ch0','ch1','ch2','ch3','ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10']].values
df_array=df_array*(float3)  # THE MULTIPLIER  

#Round the float to integers and set all the values <1 =1 
df_array_round=np.around(df_array).astype(int)
df_array_round[df_array_round<1]=1
df_values=pd.DataFrame(df_array_round)# The columns automatically get named starting 0 so we can genralize the code 

# Change back the numpy format to dataframe after the manipulation 
df_proteins=baciq[['Protein ID']]
df=pd.concat([df_proteins,df_values],axis=1)

# Separate the proteins into the two categories:- one peptide proteins, multiple peptide proteins
# Note: The current code discards one peptide proteins, we can estimate the confidence interval using the beta distribution 
df_grouped=df.groupby('Protein ID').count().reset_index().rename(columns={0:'count'})
df_onecoin=df_grouped[df_grouped['count']==1][['Protein ID', 'count']]
df_manycoins=df_grouped[df_grouped['count']!=1][['Protein ID', 'count']]

# Get the data corresponding to many peptide proteins 
df_manycoinsdata=df.merge(df_manycoins, on='Protein ID')

# Split the proteins into multiple files to parallelize 
df_manycoinsdata1=df_manycoinsdata.sort(['Protein ID'], ascending=True).reset_index().drop(['index'], axis=1)
df_manycoinsdata1['index']=np.array(xrange(len(df_manycoinsdata1)))
df_index=df_manycoinsdata1.groupby(['Protein ID']).agg({'index': [min, max]}).reset_index()
df_index['row']=np.array(xrange(len(df_index)))

divisions=len(df_index)/n_codes
spliced=df_index[divisions*(current_code_n-1):divisions*(current_code_n)]
min1=list(spliced['index']['min'])[0]
max1=list(spliced['index']['max'])[-1]
if (current_code_n==n_codes): 
    df_manycoinsdata=df_manycoinsdata1[min1:len(df_manycoinsdata1)]
else:
    df_manycoinsdata=df_manycoinsdata1[min1:max1+1]
    

# Sum up the information of rest of the channels if flag=0 
if flag==0: 
    temporary1=df_manycoinsdata[['Protein ID', int1]].rename(columns={int1:'channel1'})
    temporary2= pd.DataFrame(df_manycoinsdata.drop(['Protein ID', 'count', 'index', int1], axis=1).sum(axis=1)).rename(columns={0:'channel2'})
    df_manycoinsdata = pd.concat([temporary1, temporary2],axis=1)
    #print df_manycoinsdata.shape, temporary1.shape, temporary2.shape
else:
    temporary1=df_manycoinsdata[['Protein ID', int1, int2]]
    df_manycoinsdata= temporary1.rename(columns={int1:'channel1', int2:'channel2'})
ch1='channel1'
ch2='channel2'

######################The model and its compilation #####################################
import pystan 
code="""data {
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
sm = pystan.StanModel(model_code=code)


bins1=np.linspace(0,1,40000)
grouped_manydata= df_manycoinsdata.groupby(['Protein ID'])

flag1=0

for name, group in grouped_manydata: 
    df_temp=grouped_manydata.get_group(name)
    df_temp['sum']=df_temp[ch1]+df_temp[ch2]

    data_temp={'J':len(df_temp),# Number of peptides 
              'y':df_temp[ch1].values.tolist(), # Value of channel 1 for that peptide 
              'n':df_temp['sum'].values.tolist()}# Value of total throws for that peptide 
    fit = sm.sampling(data=data_temp, chains = 1, iter = 5005000, warmup = 5000, refresh=-1)
    sim=fit.extract()
    df_temp2= pd.DataFrame(sim["mu"]).T
    if (flag1==0):
        df_temp3=df_temp2.quantile([loconf,0.5,highconf],axis=1).T
        df_temp3['index'] = name
        final_df=df_temp3
        flag1=1
    else:
        df_temp3= df_temp2.quantile ([loconf,0.5,highconf],axis=1).T
        df_temp3['index']=name
        final_df=pd.concat([final_df, df_temp3], axis=1)

df_export=final_df
df_export.to_csv('Output/outFile_hist%d.csv'%(current_code_n))







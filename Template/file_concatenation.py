
import pandas as pd 
import numpy as np 
import sys 
n_codes=int(sys.argv[1])
i=1
flag=0 
while i<n_codes+1: 
	df_temp=pd.read_csv('Output/outFile_hist%d.csv'%(i))
	if flag==0:
		df_final=df_temp
		flag=1
	else:
		df_final=pd.concat([df_final,df_temp] , axis=0)
	i=i+1
print len(df_final)
df_final.to_csv('Output/outFile_hist.csv')

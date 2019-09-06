# BACIQ
BACIQ is the python-compatible program associated with our paper "Bayesian
Confidence Intervals for Multiplexed Proteomics Integrate Ion-Statistics
with Peptide Quantification Concordance". Please refer to the Readme.docx
for further instructions.
## Requirements
1. Unix/Linux Operating System
2. Python - 2.7.3 with
    1. Numpy - 1.13.3
    2. Pandas -  0.15.2 dev
    3. Pystan -  2.17.1.0

## Usage
1. Copy the folder `BACIQ_v1.0.zip` to the desired directory.
Extract the `BACIQ_v1.0.zip` and copy the extracted contents to the
desired directory.
2. Replace `SampleInput.csv` in the `BACIQ_v1.0` folder with your Input CSV
file. Please make sure your input CSV file has the same column names as the
`SampleInput.csv` file you are replacing. Also, input file should be in .csv
format.
3. Open Terminal on your Unix System
4. Navigate to the `BACIQ_v1.0` directory path (using `cd /path` command)
5. Use system command `ls` or `pwd` to ensure you are in the intended directory
6. Enter the following commands in Terminal to run the program:
`linux.sh`
7. Keep entering the values as asked. Ensure that the number of cores input
is not more than the total proteins in your Input file

## Output
1. Once the program finishes running, you can find then output at the following path:
`/yourpath/BACIQ_v1.0/Output/outFile_hist.csv`
Please ignore other csv files in the Output folder except `outFile_hist.csv`
2. This `outFile_hist.csv` contains:
    1. Lower end of the confidence interval 
    2. Median value 
    3. Higher end of the confidence interval
    4. Protein name as index 

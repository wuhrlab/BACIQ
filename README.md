![](https://github.com/troycomi/BACIQ/workflows/CI/badge.svg)

# BACIQ
BACIQ is the python program associated with our paper:\
["Bayesian Confidence Intervals for Multiplexed Proteomics Integrate\
Ion-Statistics with Peptide Quantification
Concordance"](https://doi.org/10.1074/mcp.TIR119.001317).

## Requirements
1. Unix/Linux Operating System
2. Python - 3.7 with
    1. pym3 - 3.7
    2. pandas -  0.25
    3. click -  7.0

## Usage
1. BACIQ can be pip installed with `pip install git+https://github.com/wuhrlab/BACIQ`
All dependencies will be installed and the program `baciq` added to path.
A virtual environment is recommended.

2. BACIQ is a command-line tool with options that can be retrieved with
```shell
$ baciq --help
Usage: baciq [OPTIONS]

Options:
  -i, --infile FILENAME   Input csv file containing counts data
  -o, --outfile FILENAME  Output file, can be csv or csv.gz
  -c1, --channel1 TEXT    Column to use as "heads" in beta-binomial
  -c2, --channel2 TEXT    Column to use as "tails" in beta-binomial or "sum"
  -s, --scaling FLOAT     Value to scale input data
  -c, --confidence FLOAT  Confidence level of quantiles to report
  -b, --bin-width FLOAT   Bin width for histogram output.  If specified,
                          confidence will be ignored
  --samples INTEGER       Number of samples for MCMC per chain
  --chains INTEGER        Number of MCMC chains
  --tuning INTEGER        Number of burn in samples per chain
  --batch-size INTEGER    Number of proteins to run at once
  --help                  Show this message and exit.
```

- infile: specify a csv file containing at a minimum the columns 'Protein ID',
  and two intensity columns.  If 'sum' is specified for channel2, all numeric
  columns will be included in the summation.  **Required.**
- outfile: specify output csv file.  If a gz file is specified the output
  will be compressed.  **Required.**
- channel1: Column name of the 'heads' observations.  Must match input column
  name exactly.  **Required.**
- channel2: Additional column to use as 'tails' observations.  If set to 'sum'
  will instead determine the sum along each row of all numeric columns, even
  if a column named 'sum' is in the input file.  **Required.**
- scaling: a value to multiply with each input value.  Prior to MCMC sampling,
  numerical values are rounded and set to a minimum of 1.  Default: 1.
- confidence: The confidence interval of quantiles to report.  E.g. if set to
  0.95, the quantiles 0.025, 0.5, and 0.975 will be reported.  Default: 0.95.
- bin-width: If specified, the output will be histograms of mu instead of
  quantiles.  Note that extremely small values will require large amounts of
  memory. Default: None.
- samples: Number of samples per MCMC chain, after tuning.  The total number
  of samples simulated will be samples * chains.  Default: 1000.
- chains: Number of independent MCMC chains to run. Default: 5.
- tuning: Number of burn-in samples to acquire before recording.  Default: 1000.
- batch-size: Number of proteins to include in each batch of MCMC runs.
  Because the BACIQ model is partially pooled, there is a minor dependence of
  output distributions based on the batch contents.  If possible, no batching
  should be used, but good performance is obtained with batch sizes of 500-1000.
  The actual batch size per run will be smaller to evenly distribute proteins
  between all batches.  Default: None

## Output
Depending on the values of `confidence` and `bin-width`, the output will be a
table of quantiles or histograms.  Each row contains information on one 
Protein ID (first column) and the columns are labeled with the quantile or bin.
The output order reflects the input sorting within batches.

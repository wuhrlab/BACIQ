#!/bin/bash
echo "Start Time: $(date)." > bash.log

int1=ch0
int2=ch1
 
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i SampleInput.csv -o Output/SampleOutput.csv
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i MedInput.csv -o Output/MedOutput.csv
time python baciq/baciq.py -c1 $int1 -c2 $int2 -i Input/BACIQ_all_proteins.csv -o Output/allProteins.csv -b 0.01 --samples 100000 --chains 5

echo "End Time: $(date)." >> bash.log

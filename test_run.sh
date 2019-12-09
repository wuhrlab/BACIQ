#!/bin/bash
echo "Start Time: $(date)." > bash.log

int1=ch0
int2=ch1
 
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i Input/SampleInput.csv -o Output/SampleOutput.csv --samples 1000 --chains 8 -b 0.1
time python baciq/baciq.py -c1 $int1 -c2 $int2 -i Input/MedInput.csv -o Output/MedOutput_bat2.csv --batch-size 3
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i Input/LargeInput.csv -o Output/LargeOutput.csv --samples 5000 --chains 4
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i Input/BACIQ_all_proteins.csv -o Output/100_unsort.csv --samples 1000 --batch-size 100

echo "End Time: $(date)." >> bash.log

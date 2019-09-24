#!/bin/bash
echo "Start Time: $(date)." > bash.log

int1=ch0
int2=ch1
 
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i SampleInput.csv -o Output/SampleOutput.csv
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i MedInput.csv -o Output/MedOutput.csv
time python baciq/baciq.py -c1 $int1 -c2 $int2 -i LargeInput.csv -o Output/LargeOutput.csv

echo "End Time: $(date)." >> bash.log

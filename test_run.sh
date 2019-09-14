#!/bin/bash
echo "Start Time: $(date)." > bash.log

int1=ch0
int2=ch1
 
float3=1
conf=0.95

n=1
mkdir "C$n"
cp -R Template/BB_code.py "C$n"/"C$n".py
python "C$n"/"C$n".py $int1 $int2 $float3 $conf

echo "End Time: $(date)." >> bash.log

cmp Output/seed1.csv Output/outFile_hist.csv

#!/bin/bash
n=1
max=1

cp -R Template/bash.log ./bash.log
echo "Start Time: $(date)." | cat - bash.log > temp && mv temp bash.log

flag=1
int1=0
int2=1
 
float3=1
conf=0.95

while [ "$n" -le "$max" ]; do
  echo "The current code in process is $n"
  mkdir "C$n"
  cp -R Template/BB_code.py "C$n"/"C$n".py
  python "C$n"/"C$n".py $n $max $flag $int1 $int2 $float3 $conf &
  sleep 20s
  n=`expr "$n" + 1`
done
wait 

python Template/file_concatenation.py $max
echo "End Time: $(date)." | cat - bash.log > temp && mv temp bash.log

cmp Output/seed1.csv Output/outFile_hist.csv

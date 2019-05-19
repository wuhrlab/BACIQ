#!/bin/bash
n=1;
read  -p "Enter the number of cores (entered value should not be more than the number of proteins) -> " max

cp -R Template/bash.log ./bash.log
echo "Start Time: $(date)." | cat - bash.log > temp && mv temp bash.log


read  -p "Enter 0 if analysing one channel against the sum of rest ; 1 if analysing two channels -> " flag 



if [ $((flag)) -eq 0 ]; then 
    read  -p "Enter the channel number starting 0 -> " int1 
    int2=999;
else 
    read  -p "Enter the first channel number starting 0 -> " int1
    read  -p  "Enter the second channel number starting 0 -> " int2
fi   
 
read -p "Enter the multiplier -> " float3
read -p "Enter the confidence interval to be calculated in fraction -> " conf 

while [ "$n" -le "$max" ]; do
  echo "The current code in process is $n"
  mkdir "C$n"
  cp -R Template/BB_code.py "C$n"/"C$n".py
  python "C$n"/"C$n".py $n $max $flag $int1 $int2 $float3 $conf &
  sleep 20s
  n=`expr "$n" + 1`;
done
wait 

python Template/file_concatenation.py $max
echo "End Time: $(date)." | cat - bash.log > temp && mv temp bash.log


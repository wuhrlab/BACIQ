#!/bin/bash
#SBATCH --output=sb.out
#SBATCH --time=0-0:20
. ~/.bashrc
conda activate baciq
echo "Start Time: $(date)." > bash.log

int1=ch0
int2=ch1
 
#baciq -c1 $int1 -c2 $int2 -i Input/SampleInput.csv -o Output/SampleOutput.csv --samples 1000 --chains 2 -b 0.1
baciq -c1 $int1 -c2 $int2 -i Input/MedInput.csv -o Output/MedOutput_bat.csv --batch-size 5
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i Input/LargeInput.csv -o Output/LargeOutput.csv --samples 5000 --chains 4
#time python baciq/baciq.py -c1 $int1 -c2 $int2 -i Input/BACIQ_all_proteins.csv -o Output/100_unsort.csv --samples 1000 --batch-size 100

# for batch in 100 500 1000 5000 10000; do
#     time python baciq/baciq.py -c1 $int1 -c2 $int2 \
#         -i Input/MedInput.csv \
#         -o /dev/null \
#         --bin-width 0.01 \
#         --samples $batch \
#         --chains 2
# done
echo "End Time: $(date)." >> bash.log

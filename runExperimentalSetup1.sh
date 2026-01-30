#!/bin/bash
#declare -a historicals=("Resilient" "Carv" "FFT")
#declare -a historicals=("Carv")
declare -a datasets=("Uber" "Twitter" "Abalone" "OnlineRetail" "Electricity")
#declare -a datasets=("OnlineRetail")
# "OnlineRetail" "Electricity")
declare -a ks=(10 20 50)

for k in "${ks[@]}"
do
        for dataset in "${datasets[@]}"
        do
                for historical in "${historicals[@]}"
                do
                        python3 src/setup1.py $dataset $historical $k
                done
        done
done
#python3 src/plotsForSetup1.py


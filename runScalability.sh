#!/bin/bash
declare -a bs=(2 4 6)
declare -a ks=(5 10 15 20 50)

for k in "${ks[@]}"
        do
        for b in "${bs[@]}"
        do
                python3 src/scalability.py $k $b
        done
done
python3 src/plotsForScalability.py



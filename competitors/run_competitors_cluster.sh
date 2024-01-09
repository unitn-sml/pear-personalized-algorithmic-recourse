#!/bin/bash

for seed in `seq 2023 2027`
do
    for model in "nn" "linear" "tree"
    do
        for quantile in "1.0" "0.25"
        do
            for dataset in "adult"  "givemecredit"
            do
                for method in "cscf" "face_cost"
                do
                    if [ -f "results/competitors/$model/validity_cost-$dataset-$method-False-300-$quantile-$corruption-$seed.csv" ]; then
                        echo "Skipping $dataset $method"
                    else
                        python competitors/cluster/submit_competitor.py --queue short_cpuQ --model $model --method $method --dataset $dataset --quantile $quantile --seed $seed --output results/competitors/$model
                    fi
                done
            done
        done
    done
done

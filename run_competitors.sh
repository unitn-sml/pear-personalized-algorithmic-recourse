#!/bin/bash

for seed in `seq 2023 2032`
do
    for model in "nn" "linear" "tree"
    do
        for quantile in "1.0" "0.25"
        do
            for dataset in "adult" "givemecredit"
            do
                for method in "fare" "efare" "mcts" "rl" "face_cost" "cscf"
                do
                    for corruption in "0.0"
                    do
                        if [ -f "results/competitors/$model/validity_cost-$dataset-$method-$quantile-300-$corruption-$seed.csv" ]; then
                            echo "Skipping $dataset $method"
                        else
                            python competitors/run_competitor.py --model $model --test-set-size 300 --method $method --corrupt-graph $corruption --dataset $dataset --quantile $quantile --seed $seed --output results/competitors/$model
                        fi
                    done
                done
            done
        done
    done
done

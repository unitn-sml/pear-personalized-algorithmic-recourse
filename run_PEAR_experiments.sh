#!/bin/bash

for dataset in "adult" "givemecredit"
do
    for model in "nn" "linear" "tree"
    do
        for quantile in "1.0" "0.25"
        do
            for seed in "2023" "2024" "2025" "2026" "2027" "2028" "2029" "2030" "2031" "2032"
            do
                for logistic in "False" "True"
                do
                    if [[ "${logistic}" == "False" ]]; then
                        logistic_command=""
                    else
                        logistic_command="--logistic-user"
                    fi
                    mpirun python interactive.py --model $model --test-set-size 300 --output results/${dataset}/$model --dataset $dataset --min-questions 10 --mcmc-steps 4 --questions 10 --seed $seed --quantile $quantile $logistic_command
                done
            done
        done
    done
done
#!/bin/bash

for dataset in "adult" "givemecredit"
do
    for choice_set_size in "2" "4"
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
                mpirun python interactive.py --choice-set-size $choice_set_size --test-set-size 300 --output results/${dataset}/$model --dataset $dataset --mcmc-steps 4 --min-questions 1 --questions 10 --seed $seed --quantile 1.0 $logistic_command
            done
        done
    done
done
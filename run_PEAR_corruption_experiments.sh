#!/bin/bash

for dataset in "adult" "givemecredit"
do
    for quantile in "1.0" "0.25"
    do
        for corruption in "0.15" "0.25" "0.5" "1.0"
        do
            for seed in "2023" "2024" "2025" "2026" "2027" "2028" "2029" "2030" "2031" "2032"
            do
                mpirun python interactive.py --corrupt-graph ${corruption} --test-set-size 300 --output results/${dataset}/${model} --dataset $dataset --min-questions 10 --mcmc-steps 4 --questions 10 --seed $seed --quantile $quantile --logistic-user
            done
        done
    done
done
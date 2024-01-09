#!/bin/bash

for s in `seq 2025 2033`
do
    python competitors/fare/train_FARE.py --model nn --dataset givemecredit --output ignore/all_fare_models --seed $s --retrain
    python competitors/fare/train_FARE.py --model nn --dataset adult --output ignore/all_fare_models --seed $s --retrain
done
#!/usr/bin/env bash
#PBS -l select=1:ncpus=50:mem=8GB
#PBS -l walltime=4:59:59
#PBS -M giovanni.detoni@unitn.it
#PBS -V
#PBS -m be

# https://github.com/open-mpi/ompi/issues/7701
export HWLOC_COMPONENTS=-gl

export PATH=$HOME/miniconda3/bin:$PATH
source ~/.bashrc

cd user-aware-recourse

conda activate aware

export PYTHONPATH=./

dataset="${D:-adult}"
model="${M:-nn}"
method="${MM:-cscf_large}"
size="${S:-300}"
output="${O:-}"
quantile="${L:-1.0}"
seed="{SEED:-2023}"

python competitors/run_competitor.py --seed "${seed}" --test-set-size "${size}" --dataset "${dataset}" --model "${model}" --method "${method}" --output "${output}" --quantile "${quantile}"
#!/usr/bin/env bash
#PBS -l select=3:ncpus=55:mem=5GB:mpiprocs=50
#PBS -l walltime=5:59:59
#PBS -M giovanni.detoni@unitn.it
#PBS -V
#PBS -m be

# https://github.com/open-mpi/ompi/issues/7701
export HWLOC_COMPONENTS=-gl

export PATH=$HOME/miniconda3/bin:$PATH
source ~/.bashrc

cd user-aware-recourse

module load openmpi-3.0.0

conda activate aware

export PYTHONPATH=./

if [[ "${L}" == "False" ]]; then
  logistic=""
else
  logistic="--logistic-user"
fi

if [[ "${XPEAR}" == "False" ]]; then
  xpear=""
else
  xpear="--xpear"
fi

if [[ "${RC}" == "False" ]]; then
  random_choice_set=""
else
  random_choice_set="--random-choice-set"
fi

choice_set="${C:-2}"
dataset="${D:-adult}"
model="${M:-nn}"
size="${S:-100}"
corrupt="${CG:-0.0}"
output="${O:-}"
quantile="${QA:-1.0}"
seed="${SEED:-2023}"

mpirun python3 interactive.py --seed "${seed}" --test-set-size "${size}" --questions "${Q}" --choice-set-size "${choice_set}" --dataset "${dataset}" --model "${model}" --corrupt-graph "${corrupt}" --quantile "${quantile}" --output "${output}" ${logistic} ${random_choice_set} ${xpear}

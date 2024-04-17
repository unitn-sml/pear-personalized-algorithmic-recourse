# Personalized Algorithmic Recourse with Preference Elicitation

This repository contains the code to replicate the experiments of the paper "De Toni, Giovanni, et al. "Personalized Algorithmic Recourse with Preference Elicitation." Transactions on Machine Learning Research (2023)." (see https://openreview.net/forum?id=8sg2I9zXgO)

### Setup and Install

PEAR code runs on Python 3.7. The instructions which follows uses `conda` as the environment manager. In theory, the code should also work for Python >= 3.7, but no testing was performed, so feel free to open an issue in case of bugs with more recent versions of Python.

```bash
# First, we create the environment
conda create --name aware python=3.7
conda activate aware

# Then, we install the various dependencies
pip install -r requirements.txt

# Lastly, we install the third-party library needed for PEAR
# This library contains the WFARE implementation
pip install git+https://github.com/unitn-sml/recourse-fare.git@v0.2.0
```

### Empirical Evaluation

The directory `results/` contains the evaluation results for all datasets and competitors. It is organized as follow:
* `adult/` and `givemecredit/`: contains PEAR evaluations for the Adult and GiveMeSomeCredit datasets, respectively, divided by the class of blackbox used (e.g., `nn` as neural network, `linear` as linear classifier and `tree` as decision tree). We used these data to generate Figure 3 and Table 1 results (and the Appendix evaluations).
* `adult_corruption/` and `givemecredit_corruption/`: contains PEAR evaluation for a missrepresented causal graph (both for Adult and GiveMeSomeCredit respectively). For this experiment, we only considered a neural network as black-box model. We used the data for Table 2.
* `competitors`: contains all competitors evaluations for the Adult and GiveMeSomeCredit datasets, respectively, divided by the class of blackbox used. We used the data to generate Table 2 and Table 4 (Appendix).
* `adult_xpear/` and `givemecredit_xpear/`: contains evaluations for the explainable version of PEAR, which are presented in Table 5 (Appendix).

In order to generate the various plots and tables used in the paper, follow the underlying commands:
```bash
conda activate aware

# Generate regret evaluation (Figure 3)
python analytics/plot_regret.py results/adult/nn results/givemecredit/nn

# Generate latex code for Table 1 (and Table 5 in the Appendix)
python analytics/plot_competitor.py results/competitors/nn

# Generate latex code for Table 2
python analytics/plot_corruption.py results/adult_corruption/nn
python analytics/plot_corruption.py results/givemecredit_corruption/nn

# Generate latex code for Table 4 (Appendix)
python analytics/plot_competitor.py results/competitors/tree
python analytics/plot_competitor.py results/competitors/linear

# Generate figures in the Appendix
python analytics/plot_appendix_results.py results/competitors/nn
```

### Run PEAR on a test case

You can run PEAR locally on your machine by using the `interactive.py` script. We use MPI to parallelize the execution over multiple core. For example, a possible invocation of the command is the following:
```bash
conda activate aware

mpirun -n 4 python interactive.py --model nn --test-set-size 300 --output . --dataset adult --mcmc-steps 4 --questions 10
```
Here, we run PEAR over 300 individuals classified negatively from the `adult` dataset. We also cap the maximum number of questions to 10 and the maximum number of MCMC sampling steps to 4. We parallelize the computation over 4 cores. More information about the arguments can be found by running `python interactive.py --help`.

### Reproduce the complete experiments

In order to run yourself the full experiments please have a look at the following scripts:
* `run_PEAR_experiments.sh`: reproduce all the PEAR experiments (Table 1, 4 and 5)
* `run_PEAR_regret_experiments.sh`: reproduce all the PEAR experiments concerning the regret (Figure 3)
* `run_PEAR_corruption_experiments.sh`: reproduce all the PEAR experiments concerning the corruption of the causal graph (Table 2)
* `run_competitors.sh`: reproduce all the evaluations for all the competitors (Table 1, Table 4)

Bear in mind, it takes **quite some time** to replicate all the results. We suggest parallelizing the execution to speed up the computations (have a look at the `cluster/` and `competitors/cluster` directories for an example).

## How to cite

Please, use the following entry to cite our work:

```
@article{
    detoni2024personalized,
    title={Personalized Algorithmic Recourse with Preference Elicitation},
    author={Giovanni De Toni and Paolo Viappiani and Stefano Teso and Bruno Lepri and Andrea Passerini},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2024},
    url={https://openreview.net/forum?id=8sg2I9zXgO},
    note={}
}
```








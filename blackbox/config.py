from blackbox.givemecredit.preprocessing_utils import fix_give_me_credit

import numpy as np

# Dataset specific configurations
DATA_CONFIG = {
    "adult": {
        "dataset": "blackbox/adult/raw/data.csv",
        "target": "income_target",
        "target_negative": "<=50K",
        "drop": ["fnlwgt", "education_num", "predicted"],
        "immutables": ["age", "sex", "native_country", "relationship", "race", "marital_status"],
        "preprocessing": None
    },
    "givemecredit": {
        "dataset": "blackbox/givemecredit/raw/cs-training.csv",
        "target": "SeriousDlqin2yrs",
        "target_negative": 1,
        "drop": [],
        "immutables": ["NumberOfDependents", "age"],
        "preprocessing": fix_give_me_credit
    },
    "synthetic": {
        "dataset": "blackbox/synthetic/raw/train.csv",
        "target": "loan",
        "target_negative": "bad",
        "drop": [],
        "immutables": ["age", "sex", "country", "credit"],
        "preprocessing": None
    }
}

# Define the param grid for gridsearch
PARAM_GRID = {
    "svc": [
        {'C': np.linspace(1, 10, num=5), 'gamma': ['auto'], 'kernel': ['rbf']}
    ],
    "tree": [
        {'criterion': ["gini", "entropy"],
            'splitter': ["best", "random"]}
    ],
    "linear": [
        {
        }
    ]
}

# Specific configuration for the neural network
# The parameters here give us a reasonable accuracy. We did some
# hyperparameter tuning manually, but we stopped once we got a
# decent f1 score and accuracy tradeoff.
NN_CONFIG = {
    "adult": {
        'nn': {
            "iterations": 15,
            "layers": 3,
            "batch_size": 1024
        },
        "linear": {
            "iterations": 15,
            "layers": 1,
            "batch_size": 1024
        }
    },
    "givemecredit": {
        'nn': {
            "iterations": 15,
            "layers": 3,
            "batch_size": 1024
        },
        "linear": {
            "iterations": 15,
            "layers": 1,
            "batch_size": 1024
        }
    },
    "synthetic": {
        'nn': {
            "iterations": 5,
            "layers": 3,
            "batch_size": 1024
        },
        "linear": {
            "iterations": 35,
            "layers": 1,
            "batch_size": 1024
        }
    }
}


MIXTURE_MEAN_LIST = {
    "adult": [[74, 26, 93, -7, 66, -25, 65, 27, 1, 92, -10, 97, -13, -17, 26, -1, -56, 69, 90, -32, 82, 87, -56, -32, -36, -47, 78, 89, 39, -54, 85, 5], [14, 55, 45, -50, -58, -43, 7, 29, 52, -13, 14, 31, -56, 82, 87, -38, -5, 36, 57, 30, 74, 99, 73, 20, 10, 97, 61, 66, 57, -25, 99, 33], [52, 47, 86, -40, 52, -56, -37, -43, 47, 23, 34, 21, 59, 89, 22, -18, 61, -9, 60, -33, 32, -56, -27, 67, 90, 53, -28, 66, 43, 34, 10, 49], [-11, 69, 64, 97, 52, -54, 22, 85, 88, 64, 36, 34, 7, 63, -11, 7, -24, 6, -5, -1, -27, 18, 60, -29, 7, 56, -59, -15, 81, 57, 84, -40], [34, -58, 69, 93, 95, 1, 33, 84, -47, -33, 17, 75, -23, -40, -50, -29, -35, 95, 55, 91, 5, -16, -31, 68, 60, 41, 76, -30, 79, -37, 11, 1], [45, 15, 20, 60, -57, -24, 67, 76, -32, 39, 64, 22, 40, 33, 60, -44, 13, -58, -35, 7, -33, 96, -34, -45, 29, -13, 69, 12, 68, 99, -24, 80]],
    "givemecredit": [[76, 52, 93, -3, 52, 48, 37, -49, -43, 57, 20, 73, -17, -49, 86, -4, 9, 30, 2, 86, 18, 84, 93, -6, 29, -34, 85, 99, 1], [36, 27, 83, -51, 39, 88, 86, 77, -36, 47, 45, -32, 80, 52, -15, 15, -70, 9, 56, -6, 47, 88, -68, 40, -65, -60, 98, -65, -31], [12, 7, -11, 76, 47, -45, 87, 47, 71, 46, -44, -19, -42, 58, -21, 89, 88, -54, 9, 79, 10, -2, 16, -65, 1, 87, -3, 98, 51], [76, 6, -66, 56, 88, 68, 34, -14, 69, -13, 99, -47, 58, 8, -57, 80, -8, 97, 97, -38, 9, -22, 25, -27, -54, 52, -34, 85, 69], [82, 23, -68, 65, 41, 64, 95, -35, 42, 2, 87, 7, 95, 30, 40, 68, 22, 50, -45, -38, 64, -59, 85, 96, -39, 33, -58, -9, -16], [-28, -29, 94, -56, 57, -16, 70, 22, 81, 81, 59, 82, 50, 78, 98, 41, 14, -30, 73, 17, -32, -5, 43, 48, 43, 71, 22, 92, 37]],
    "synthetic": [[21, 67, 10, -63, 93, 78, -26, 38, 68, 67, 91, -23, -66, 11, -12, -9, 87, -57, 40, 70, 13, -19, 18, 6, -27, -21], [28, -50, 83, 47, 48, 85, -2, 35, -17, -61, -45, 68, 73, 35, 98, 93, 30, 68, 10, -38, -41, 45, 73, 5, -22, 23], [-20, 27, 2, -37, 85, 81, -3, -42, -13, -5, 24, 97, 83, 30, -37, 32, 72, 24, 10, -60, -40, -33, 80, -14, -60, 23], [75, 95, 68, -48, -26, 85, 80, 74, 95, 62, 34, 81, 78, -54, 75, 36, -29, -26, 87, 3, 36, 65, 75, 99, 87, 58], [46, 79, -20, 84, 63, -13, 27, -4, 63, -61, 93, 88, -26, 28, 91, 13, -68, 48, 74, 70, -59, -67, 32, 69, 76, 48], [84, 96, 94, 6, 51, -70, 60, 89, 69, 82, 43, 75, 56, 2, 41, 74, -26, -58, 84, 52, -45, -30, 5, 42, 96, -25]]
}

WFARE_CONFIG_MCTS = {
    "adult": {
        0.00: 1,
        0.15: 1e-5,
        0.25: 1e-5,
        0.5: 1e-5,
        1.00: 1e-5,
    },
    "givemecredit": {
        0.00: 1,
        0.15: 1e-5,
        0.25: 1e-5,
        0.5: 1e-5,
        1.00: 1e-5,
    }
}

WFARE_CONFIG = {
    "adult": {
        "environment": "blackbox.adult.mock_adult_env.AdultEnvironment",
        "training_steps": 500,
        "batch_size": 50,
        "buffer_size": 200,
        "expectation": True,
        "policy_config": {
            "observation_dim": 108,
            "encoding_dim": 54,
            "hidden_size": 54
        },
        "mcts_config": {
            "exploration": True,
            "number_of_simulations": 35,
            "dir_epsilon": 0.3,
            "dir_noise": 0.3,
            "action_cost_coeff": 1.0
        }
    },
    "givemecredit": {
        "environment": "blackbox.givemecredit.givemecredit_env.GiveMeCreditEnv",
        "training_steps": 700, #1500,
        "batch_size": 50,
        "buffer_size": 200,
        "sample_hard": 0.3,
        "expectation": True,
        "finetune": True,
        "policy_config": {
            "observation_dim": 18,
            "encoding_dim": 9,
            "hidden_size": 9,
            "learning_rate": 0.003
        },
        "mcts_config": {
            "exploration": True,
            "number_of_simulations": 35,
            "dir_epsilon": 0.3,
            "dir_noise": 0.3,
            "action_cost_coeff": 1.0
        }
    },
    "synthetic": {
        "environment": "blackbox.synthetic.synthetic_env.SyntheticEnv",
        "training_steps": 800,
        "batch_size": 50,
        "buffer_size": 200,
        "expectation": True,
        "policy_config": {
            "observation_dim": 30,
            "encoding_dim": 18,
            "hidden_size": 18,
            "action_cost_coeff": 0.7
        },
        "mcts_config": {
            "exploration": True,
            "number_of_simulations": 25,
            "dir_epsilon": 0.3,
            "dir_noise": 0.3
        }
    },
}

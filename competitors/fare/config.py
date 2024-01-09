FARE_CONFIG_MCTS = {
    "adult": {
        0.00: 1,
        0.15: 1e-4,
        0.25: 1e-4,
        0.5: 1e-4,
        1.00: 1e-6,
    },
    "givemecredit": {
        0.00: 1,
        0.15: 1e-4,
        0.25: 1e-4,
        0.5: 1e-4,
        1.00: 1e-5,
    }
}

FARE_CONFIG = {
    "adult": {
        "environment": "blackbox.adult.mock_adult_env.AdultEnvironment",
        "training_steps": 500,
        "batch_size": 50,
        "policy_config": {
            "observation_dim": 102,
            "encoding_dim": 51,
            "hidden_size": 51
        },
        "mcts_config": {
            "exploration": True,
            "number_of_simulations": 15,
            "dir_epsilon": 0.3,
            "dir_noise": 0.3
            }
    },
    "givemecredit": {
        "environment": "blackbox.givemecredit.givemecredit_env.GiveMeCreditEnv",
        "training_steps": 500,
        "batch_size": 50,
        "policy_config": {
            "observation_dim": 10,
            "encoding_dim": 5,
            "hidden_size": 5
        },
        "mcts_config": {
            "exploration": True,
            "number_of_simulations": 15,
            "dir_epsilon": 0.3,
            "dir_noise": 0.3
        }
    },
    "synthetic": {
        "environment": "blackbox.synthetic.synthetic_env.SyntheticEnv",
        "training_steps": 500,
        "batch_size": 50,
        "policy_config": {
            "observation_dim": 25,
            "encoding_dim": 18,
            "hidden_size": 18
        },
        "mcts_config": {
            "exploration": True,
            "number_of_simulations": 15,
            "dir_epsilon": 0.3,
            "dir_noise": 0.3
            }
    },
}
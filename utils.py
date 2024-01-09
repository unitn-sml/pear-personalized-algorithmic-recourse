from recourse_fare.models.WFAREFiner import WFAREFiner

def build_wfarefiner(wfare_model, fare_model, policy_config: dict):

    tmp =  WFAREFiner(
        fare_model, wfare_model.model, policy_config, wfare_model.environment_config, wfare_model.mcts_config,
        batch_size=wfare_model.batch_size, training_buffer_size=wfare_model.training_buffer_size,
        validation_steps=wfare_model.validation_steps, expectation=wfare_model.expectation
    )
    tmp.policy = wfare_model.policy
    return tmp

def filter_negative_classes(X, W_test, mixture_classes, quantile=1.0, num_examples=300):

    X = X[X.predicted <= 0.5]
    quantile = X.predicted.quantile(quantile)
    original_model_score = X["predicted"]
    X = X[(X.predicted <= quantile)]
    print("[*] HARD EXAMPLES: ", len(X))

    # We sample the examples for the evaluation
    X = X.sample(num_examples)

    # We save the original model scores
    original_model_score = X["predicted"]
    X.drop(columns="predicted", inplace=True)

    W_test = W_test.iloc[X.index]
    mixture_classes = mixture_classes.iloc[X.index]

    X.reset_index(inplace=True, drop=True)

    W_test.reset_index(inplace=True, drop=True)
    mixture_classes.reset_index(inplace=True, drop=True)
    original_model_score.reset_index(inplace=True, drop=True)

    return X, W_test, mixture_classes, original_model_score
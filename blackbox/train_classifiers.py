from recourse_fare.utils.preprocess.fast_preprocessor import FastPreprocessor, StandardPreprocessor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score

from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd

from argparse import ArgumentParser
import random

import dill as pickle

import torch

from utils_blackbox.trainer_torch import train_nn
from blackbox.config import NN_CONFIG, DATA_CONFIG, PARAM_GRID

if __name__ == "__main__":

    # Initialize the parser
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="adult", help="Dataset name (adult, givemecredit)")
    parser.add_argument("--model", type=str, default="svc", help="Model type we want to train (svc, tree, nn, linear)")

    args = parser.parse_args()

    # Seed for reproducibility
    np.random.seed(2023)
    torch.manual_seed(2023)
    random.seed(2023)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Extract the right information from the available configs
    target = DATA_CONFIG.get(args.dataset).get("target")
    target_bad_value = DATA_CONFIG.get(args.dataset).get("target_negative")
    columns_to_drop = DATA_CONFIG.get(args.dataset).get("drop")
    data_path = DATA_CONFIG.get(args.dataset).get("dataset")
    preprocessing_step = DATA_CONFIG.get(args.dataset).get("preprocessing")

    # Read data and preprocess them
    X = pd.read_csv(data_path)
    X.dropna(inplace=True)

    y = X[target].apply(lambda x: 0 if x==target_bad_value else 1)
    X.drop(columns=[target], inplace=True)

    # We drop some columns we consider hardly interpretable or duplicates.
    X.drop(columns=columns_to_drop, inplace=True)

    # Apply some custom preprocessing to further clean the data
    if preprocessing_step:
        X = preprocessing_step(X, y)
        y = y.loc[X.index]
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)

    # Split the dataset into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)

    # Reset the index
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # Save the training/testing datasets
    X_train.to_csv(f"blackbox/{args.dataset}/train_data_{args.dataset}.csv", index=None)
    X_test.to_csv(f"blackbox/{args.dataset}/test_data_{args.dataset}.csv", index=None)

    # Train the preprocessor
    preprocessor = FastPreprocessor()
    preprocessor.fit(X_train)

    # Model selection with grid search
    if args.model == "svc":
        blackbox_model = SVC(class_weight="balanced", probability=True)
    elif args.model == "tree":
        blackbox_model = DecisionTreeClassifier(class_weight="balanced", max_depth=len(X_train.columns))
    elif args.model == "linear":
        blackbox_model = LogisticRegression(class_weight="balanced", penalty='none')

    # If we are using a tree/linear model, the user gridsearch to find the best parameters
    if args.model != "nn":
        best_model = GridSearchCV(blackbox_model, PARAM_GRID.get(args.model),
                                scoring="f1", n_jobs=-1,
                                verbose=3)        
        # Run the gridsearch
        best_model.fit(preprocessor.transform(X_train), y_train)
    else:

        # Train the neural network
        best_model = train_nn(
            preprocessor.transform(X_train, type="dataframe"), y_train.copy(),
            preprocessor.transform(X_test, type="dataframe"), y_test.copy(),
            linear=(args.model == "linear"),
            iterations=NN_CONFIG.get(args.dataset).get(args.model).get("iterations"),
            layers=NN_CONFIG.get(args.dataset).get(args.model).get("layers", 0),
            batch_size=NN_CONFIG.get(args.dataset).get(args.model).get("batch_size")
        )

    # Evaluate the model and print the classification report for the two classes
    if args.model != "nn":
        output = best_model.predict(preprocessor.transform(X_test))
    else:
        with torch.no_grad():
            best_model.eval()
            output = best_model(
                torch.FloatTensor(preprocessor.transform(X_test))
            )
            output = torch.round(output).flatten().numpy()        
    
    print(classification_report(output, y_test))
    print(f1_score(output, y_test))

    # Save the best estimator to disk
    if args.model != "nn":
        with open(f"blackbox/{args.dataset}/model_{args.model}_{args.dataset}.pth", "wb") as model_file:
            pickle.dump(
                best_model.best_estimator_, model_file
            )
    else:
        torch.save(best_model.state_dict(),
                   f"blackbox/{args.dataset}/model_{args.model}_{args.dataset}.pth")
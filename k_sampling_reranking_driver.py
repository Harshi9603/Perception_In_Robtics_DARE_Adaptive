import os
import csv
import time

os.environ["RAY_DEDUP_LOGS"] = "0" # Remove dedup log

import ray
import dill
import hydra
import torch
import numpy as np

from test_parameter import *
from test_worker import TestWorker
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import optuna 

from test_driver import run_test

from k_sampling_reranking_helper import k_sampling_reranking_instance

def main(number_of_trials: int) -> None: 
    
    ray.init()
    # init optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=number_of_trials)

    print("best value:", study.best_value)
    print("best params:", study.best_params)
    # Save all trials
    df = study.trials_dataframe()
    # save with data and time in title
    os.makedirs("optuna", exist_ok=True)
    df.to_csv(
        os.path.join("optuna", f'optuna_trials_{time.strftime("%d_%m_%Y_%H_%M_%S")}.csv'),
        index=False
    )

    print("best trial number:", study.best_trial.number)
    print("best trial attrs:", study.best_trial.user_attrs)
        
def objective(trial):
    # Suggest parameters
    a = trial.suggest_float("a", 0.0, 10.0)
    b = trial.suggest_float("b", 0.0, 10.0)
    c = trial.suggest_float("c", 0.0, 10.0)
    d = trial.suggest_float("d", 0.0, 10.0)
    e = trial.suggest_float("e", 0.0, 10.0)
    f = trial.suggest_float("f", 0.0, 10.0)
    g = trial.suggest_float("g", 0.0, 10.0)
    
    k = trial.suggest_int("k", 1, 10)

    # Run your pipeline with those parameters
    k_sampling_reranking_instance.set_parameters(a, b, c, d, e, f, g, k)
    result_csv = run_test()
    
    # open csv and average the column "Travel Distanc"
    with open(result_csv, mode='r') as file:
        reader = csv.DictReader(file)
        travel_distances = [float(row['Travel Distance']) for row in reader]
        average_travel_distance = sum(travel_distances) / len(travel_distances)

    # Return the metric you want to optimize
    return average_travel_distance
    

if __name__ == '__main__':
    number_of_trials = 10
    main(number_of_trials)
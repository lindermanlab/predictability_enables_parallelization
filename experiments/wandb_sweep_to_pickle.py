"""
wandb_sweep_to_pickle.py

helper file to save out wandb sweeps to pickle files

Usage:

name of sweep: user/project/sweep_id
name of pickle file: a helpful name for the pickle file

Note: very annoyingly, wandb current version (0.22.2, as of this time) of wandb is buggy
Make sure to use version 0.21.4 of wandb
See discussion: https://github.com/wandb/wandb/issues/10647
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from tqdm import tqdm
import pandas as pd
import argparse
import wandb
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True, help="Wandb sweep id")
    parser.add_argument("--name", type=str, required=True, help="Name of pickle file")
    parser.add_argument(
        "--date", type=str, required=True, help="Directory to save files"
    )
    parser.add_argument("--project", type=str, help="wandb project")
    args = parser.parse_args()

    api = wandb.Api()
    project = args.project
    saved_files = "./saved_files"
    os.makedirs(saved_files, exist_ok=True)
    sweep = api.sweep(f"{project}/{args.sweep_id}")
    dataframes = []
    for run in tqdm(sweep.runs):
        history = []
        run.load()
        for row in run.scan_history():  # important to get all the run information out
            history.append(row)
        run_data = pd.DataFrame(history)
        for key, value in run.config.items():
            run_data[key] = value  # Add configuration as columns
        dataframes.append(run_data)

    df = pd.concat(dataframes, ignore_index=True)
    print(df.shape)

    df.to_pickle(f"{saved_files}/{args.date}_{args.name}_{args.sweep_id}.pkl")

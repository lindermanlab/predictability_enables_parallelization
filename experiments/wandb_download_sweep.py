"""
wandb_download_script.py
Helper script to get pickles from different seeds into the same directory
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import wandb
import os
import argparse

def download_wandb_files(name, date, sweep_id, project):
    api = wandb.Api()

    # Make subdirectory
    saved_files = "./saved_files"
    save_dir = f"{saved_files}/{date}_{name}_{sweep_id}/"
    os.makedirs(save_dir, exist_ok=True)

    # Get all runs under that sweep
    sweep = api.sweep(f"{project}/{sweep_id}")
    runs = sweep.runs

    for run in runs:
        # Check if there are any files (even if run failed, sometimes artifacts still exist)
        for file in run.files():
            if file.name.endswith(".pkl"):
                print(f"Downloading {file.name} from run {run.name}...")
                file.download(
                    root=save_dir, replace=True
                )  # downloads into current directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download files from wandb")
    parser.add_argument("--name", type=str, help="experiment name")
    parser.add_argument("--sweep_id", type=str, help="Wandb sweep ID")
    parser.add_argument("--date", type=str, help="Directory to save files")
    parser.add_argument("--project", type=str, help="wandb project")
    args = parser.parse_args()
    download_wandb_files(args.name, args.date, args.sweep_id, args.project)
    print("Done")

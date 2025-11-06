# How to replicate our experiments

We use `wandb` and `hydra` to run the experiments. 

Each experiment is run by an appropriately named .py file (eg `fig2.py`), which has a corresponding hydra config in `configs/`. The particular sweep used for each figure is specified in the correspoding yaml file in `wandb_sweep_configs/`. The sweep can then be run on a SLURM cluster with `wandb_sweep.sh`

The wandb logs for our paper can be found [here](https://wandb.ai/xavier_gonzalez/predictability/sweeps). 

We provide two helper functions for extracting results from wandb sweeps:
* `wandb_sweep_to_pickle.py`: downloads a sweep and pickles it.
* `wandb_download_sweep.py`: if every one created a pickle file, this script downloads all of the pickle files and puts them in the same folder.

Finally, we include our plotting code in `plotting_code/`.

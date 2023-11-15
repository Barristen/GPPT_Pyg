# GPPT

This repo contains code accompanying the paper, 	[GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks (Mingchen Sun et al., KDD 2022)](https://dl.acm.org/doi/abs/10.1145/3534678.3539249).

### Dependencies(same as https://github.com/sheldonresearch/ProG)
This code requires the following:
* python 3.9+
* PyTorch v2.0.1
* torch-geometric  2.3.1

### Data
We evaluate our model on Citeseer, see the usage instructions in `utils.py` .

### Hyperparameters
The hyperparameters settings see `get_args.py`.

### Usage
To run the code, see the usage instructions at the top of `GPPT_Pyg.py`.


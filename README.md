# Decision Tree Creator

## Requirements

This project requires Python v3.6.x and pip3 to run.

## Setup

Requirements:
- Python 3
- pip3

1. Install dependencies

Dependency management is done through pip.

`pip3 install -r requirements.txt`

2. Configuring tree

You can change the hyperparameters explored as well as which dataset you wish to use in `config.py`. Should you want to use your own data, all you have to do is include it in the root directory named as `<dataset_name>_dataset.txt` and then change the DATASET variable in config.py to <dataset_name>.

## Available Scripts

From the root directory, you can run:

`python3 -m full_run` to run the decision tree creation.

Otherwise you can use the Jupyter Notebook `tree.ipynb` included in the root directory and follow the instructions on there.


Note that to visualise the trees you will need to use the jupyter notebook.


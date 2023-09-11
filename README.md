# UMIM
Code for the Paper "Online Task-Agnostic Continual Learning through Unsupervised Mutual Information Maximization"


## Requirements
see requirements.txt for a full list

## Run Supervised Learning Experiments
For each setup, there is a separate file, e.g., run_cifar10_experiments.py. 

### Mixed Sequence Data
Note that data for the mixed sequence experiments must be downloaded prior to running the experiments and put into the following folder structure: \

test_meta_path = "celeb_data/test/all_data_iid_01_05_keep_5_test_9.json" \
train_meta_path = "celeb_data/train/all_data_iid_01_05_keep_5_train_9.json"\

The file build_celeba.py will take care of loading and preparing the files.

### Tiny ImageNet Data

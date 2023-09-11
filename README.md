# UMIM
Code for the Paper "Online Task-Agnostic Continual Learning through Unsupervised Mutual Information Maximization"


## Requirements
see ```requirements.txt``` for a full list

## Run Supervised Learning Experiments
For each setup, there is a separate file, e.g., ```run_cifar10_experiments.py```. 

### Mixed Sequence Data
Note that data for the mixed sequence experiments must be downloaded prior to running the experiments and put into the following folder structure:

```
./celeb_data/test/all_data_iid_01_05_keep_5_test_9.json
./celeb_data/train/all_data_iid_01_05_keep_5_train_9.json
```

The file ```build_celeba.py``` will take care of loading and preparing the files.

### Tiny ImageNet Data
Get data here: https://www.kaggle.com/c/tiny-imagenet and load into the following folder structure within the main repo folder:
```
./tiny-imagenet-200/train/*/images/*.JPEG
./tiny-imagenet-200/val/val_annotations.txt
./tiny-imagenet-200/val/images/
./tiny-imagenet-200/wnids.txt
./tiny-imagenet-200/words.txt
```
The file ```build_tinyimage.py``` will take care of loading and preparing the files. 

## Run Reinforcement Learning Experiments

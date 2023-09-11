# UMIM
Code for the Paper "Online Task-Agnostic Continual Learning through Unsupervised Mutual Information Maximization"

### Abstract
Catastrophic forgetting remains a challenge for artificial learning systems, especially in the case of Online learning, where task information is unavailable. This work proposes a novel task-agnostic approach for class-incremental Continual Learning that combines expansion and regularization methods through a non-parametric model. The model adds new experts automatically while regularizing old experts under a variational Bayes paradigm. Expert selection is implemented using Contrastive Learning techniques, catastrophic forgetting is mitigated by minimizing mutual information between the experts' posterior and prior feature embeddings. Importantly, our method successfully handles single source tasks such as split-MNIST and split-CIFAR-10/100, mixed source tasks like split-CIFAR-100/split-F-CelebA, and long task sequences such as split-TinyImageNet, without using generative models or replay mechanisms. We also successfully extend our approach to Continual Reinforcement learning tasks. Our method achieves these results in a task-agnostic and replay-free setting, making it more flexible than most existing Continual Learning approaches without compromising performance.


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
Simply run the file ```SAC_main.py```. All RL algorithms are implemented on top of Soft-Actor-Critics, hence the files are named ```SAC_*.py```.

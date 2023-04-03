from models import ResNet18
from medmnist import INFO
import torch
import torch.nn as nn
ds_name = "breastmnist"
config = {
    "ds_name": ds_name,
    "seed": 1,  # random seed
    "device": "cuda" if torch.cuda.is_available() else 'cpu',
    "model": ResNet18,  # the model to be trained the p ps and pt are only relevant in the fedreg.
    "algorithm": "fedreg",  # FL optimizer, can be FedAvg, FedProx, FedCurv or SCAFFOLD
    "n_classes": len(INFO[ds_name]["label"]),
    "n_channels": INFO[ds_name]["n_channels"],
    "task": INFO[ds_name]["task"],
    "data_path": "./data" if True else "..",
    "num_clients": 2,
    "participation_percent": 1,
    "global_epochs": 3,
    "local_epochs": 2,  # the number of epochs in local training stage
    "batch_size": 10,  # the batch size in local training stage
    "log_path": "logs",  # the path to save the log file
    "train_transform": None,  # the preprocessing of train data, please refer to torchvision.transforms
    "test_transform": None,  # the preprocessing of test dasta
    "eval_train": True,  # whether to evaluate the model performance on the training data. Recommend to False when the training dataset is too large
    "gamma": 0.3,  # the value of gamma when FedReg is used, the weight for the proximal term when FedProx is used, or the value of lambda when FedCurv is used
    "iid": True,
    "criterion":   nn.BCEWithLogitsLoss if INFO["bloodmnist"]["task"] == "multi-label, binary-class" else nn.CrossEntropyLoss,
    "learning_rate": 0.1,
    "mu":0.1
}
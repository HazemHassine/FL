import torch.nn as nn
from models import CNN
config = {'baseline': True,
 'batch_size': 32,
 'criterion': nn.CrossEntropyLoss,
 'data_path': './data',
 'device': 'cpu',
 'ds_name': 'pneumoniamnist',
 'evaluate': True,
 'model': CNN,
 'learning_rate': 0.1,
 'local_epochs': 20,
 'log_path': './logs',
 'n_channels': 1,
 'n_classes': 2,
 'seed': 1,
 'task': 'binary-class',
 'test_transform': None,
 'train_transform': None
}
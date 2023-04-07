from models import CNN
import torch.nn as nn

config = {'algorithm': 'fedavg',
          'baseline': False,
          'batch_size': 10,
          'criterion': nn.CrossEntropyLoss,
          'data_path': './data',
          'device': 'cpu',
          'ds_name': 'organcmnist',
          'eval_train': True,
          'gamma': None,
          'global_epochs': 4,
          'iid': False,
          'learning_rate': 0.01,
          'local_epochs': 4,
          'log_path': './logs',
          'model': CNN,
          'n_channels': 1,
          'n_classes': 11,
          'num_clients': 3,
          'participation_percent': 1.0,
          'seed': 1,
          'task': 'multi-class',
          'test_transform': None,
          'train_transform': None,
          'CV': True,
          'k': 5
          }

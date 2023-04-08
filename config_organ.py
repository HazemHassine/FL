from models import CNN
import torch.nn as nn

config = {'algorithm': 'fedreg',
          'baseline': False,
          'batch_size': 50,
          'criterion': nn.CrossEntropyLoss,
          'data_path': './data',
          'device': 'cpu',
          'ds_name': 'organcmnist',
          'eval_train': True,
          'gamma': None,
          'global_epochs': 4,
          'iid': False,
          'learning_rate': 0.01,
          'local_epochs': 3,
          'log_path': './logs',
          'model': CNN,
          'n_channels': 1,
          'n_classes': 11,
          'num_clients': 2,
          'participation_percent': 1.0,
          'seed': 1,
          'task': 'multi-class',
          'test_transform': None,
          'train_transform': None,
          'ps_eta': 0.1,
          'pt_eta': 0.001,
          'p_iters': 4, 
          'mu': 0.1,
          'gamma': 0.4

          }

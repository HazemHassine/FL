from typing import Tuple, OrderedDict
from torch.utils.data import DataLoader
from utils import CustomDataset
import torch
import numpy as np
from torch.optim import SGD

class NormalClient():
    def __init__(self, id, config, train_dataset, test_dataset, data_idxs, test_idxs) -> None:
        self.id = id
        self.train_loader = DataLoader(CustomDataset(train_dataset, data_idxs),batch_size=config["batch_size"], shuffle=True)
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(CustomDataset(test_dataset, test_idxs), batch_size=32, shuffle=False)
        self.model = config["model"](config["n_channels"], config["n_classes"])
        self.device = config["device"]
        self.len_test = len(test_dataset)
        self.config = config
        self.inner_optimizer = SGD
        self.loss_fn = config["criterion"]()
        self.num_train_samples = len(data_idxs)

    def train(self):
        optimizer = SGD(self.model.parameters(), self.config["learning_rate"])
        self.model.train()
        train_loss = []
        for l_epoch in range(self.config["local_epochs"]):
            for x, y in self.train_loader:
                x.to(self.device)
                y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = self.loss_fn(logits, y.squeeze(-1))
                loss.backward()
                optimizer.step()
                train_loss.append(loss.detach().item())
        print(np.mean(train_loss))
        return self.num_train_samples, self.model.state_dict()

    def get_param(self) -> OrderedDict:
        return self.model.state_dict()

    def set_param(self) -> bool:
        self.model.load_state_dict()
        return True

    def test(self) -> Tuple[int, int]:
        self.model.eval()
        total_correct = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x.to(self.device)
                y.to(self.device)
                logits = self.model(x)
                preds = np.argmax(logits, dim=1)
                correct = np.sum(preds == y)
            total_correct += correct
        return total_correct, self.len_test
    
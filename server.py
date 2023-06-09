from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
from functools import partial
import random
from tqdm import tqdm
import copy
import torch
from utils import CustomDataset
import os


class Server:
    def __init__(self, config, data_dict, test_dict, train_dataset, test_dataset):
        self.config = config
        self.data_dict = data_dict
        self.test_dict = test_dict
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.global_model = config["model"](
            config["n_channels"], config["n_classes"])
        self.client_model = config["model"](
            config["n_channels"], config["n_classes"])
        self.client_ids = [*data_dict.keys()]
        self.test_loader = DataLoader(CustomDataset(
            test_dataset, list(range(len(test_dataset)))), batch_size=32)
        self.participation_percent = config["participation_percent"]
        ###### CHOOSING THE CLIENT ######
        if config["algorithm"] in ["fedavg", "fedbn"]:
            from normal_client import NormalClient
            self.client = NormalClient
        elif config["algorithm"] == "fedprox":
            from fedprox_client import FedProxClient
            self.client = partial(
                FedProxClient, config["mu"] if config["mu"] else 0.1)
        else:
            from fedreg_client import FedRegClient
            self.client = partial(FedRegClient, gamma=self.config["gamma"] if self.config["gamma"] else 0.4,
                                  ps_eta=self.config["ps_eta"] if self.config["ps_eta"] else 2e-1,
                                  pt_eta=self.config["pt_eta"] if self.config["pt_eta"] else 2e-3,
                                  p_iters=10)
        ################################
        self.make_clients()
        self.len_test = len(test_dataset)
        self.device = config["device"]
        self.data_frame = pd.DataFrame(
            columns=["Accuracy"], index=list(range(1, config["global_epochs"]+1)))

    def make_clients(self):
        self.clients = []
        for id in self.client_ids:
            self.clients.append(
                self.client(id, self.config, self.train_dataset,
                            self.test_dataset, self.data_dict[id], self.test_dict[id])
            )

    def test_locals(self):
        results = {id: [] for id in self.client_ids}
        for client in self.clients:
            results[client.id] = client.test()
        return results

    def aggregate_(self, wstate_dict):
        """
        Aggregates a dictionary of weighted state dictionaries.

        Args:
        - wstate_dict: a dictionary where the keys are ids and the values are tuples
          containing the weight and state dict for that id.

        Returns:
        - aggregated_state_dict: the aggregated state dictionary
        """

        # Make a deep copy of the first state dict to use as a starting point
        starting_id, (starting_weight, starting_state_dict) = next(
            iter(wstate_dict.items()))
        aggregated_state_dict = copy.deepcopy(starting_state_dict)

        # Iterate over the keys in the aggregated state dict
        for key in aggregated_state_dict.keys():
            # If the key is a tensor, aggregate it based on the weights
            if isinstance(aggregated_state_dict[key], torch.Tensor):
                # Initialize a tensor with zeros of the same shape and device as the original tensor
                aggregated_tensor = torch.zeros_like(
                    aggregated_state_dict[key])
                total_weight = starting_weight
                aggregated_tensor += starting_weight * starting_state_dict[key]

                # Iterate over the wstate_dict and sum the weighted tensors
                for id, (weight, state_dict) in wstate_dict.items():
                    if id == starting_id:
                        continue
                    aggregated_tensor += weight * state_dict[key]
                    total_weight += weight

                # Divide by the sum of weights to get the weighted average
                aggregated_tensor = torch.divide(
                    aggregated_tensor, total_weight)
                # Set the tensor in the aggregated state dict to the weighted average
                aggregated_state_dict[key] = aggregated_tensor

            # If the key is a nested dictionary, recursively aggregate it
            elif isinstance(aggregated_state_dict[key], dict):
                # Create a list of the corresponding sub-dictionaries from each state dict
                sub_dicts = [weighted_state_dict[1][key]
                             for weighted_state_dict in wstate_dict.values()]
                # Create a new dictionary with the sub-dictionaries and their corresponding weights
                id_to_weighted_sub_dict = {id: (
                    weighted_state_dict[0], sub_dicts) for id, weighted_state_dict in wstate_dict.items()}
                # Recursively call the aggregate_weighted_state_dicts function on the sub-dictionaries
                aggregated_sub_dict = self.aggregate_weighted_state_dicts(
                    id_to_weighted_sub_dict)
                # Set the aggregated sub-dictionary in the aggregated state dict
                aggregated_state_dict[key] = aggregated_sub_dict

            # If the key is anything else, raise an error
            else:
                raise ValueError(
                    f"Unexpected value type for key '{key}': {type(aggregated_state_dict[key])}")
        self.global_model.load_state_dict(aggregated_state_dict)
        return aggregated_state_dict

    def train(self, k=None):
        losses = []
        epochs = self.config["global_epochs"]
        for epoch in range(1, epochs+1):
            clients = random.sample(self.clients, int(
                len(self.clients) * self.participation_percent))
            states_dict = {}
            for client in clients:
                client.model.load_state_dict(self.global_model.state_dict())
                w, local_update = client.train(epoch)
                states_dict[client.id] = [w, local_update]
            self.aggregate_(states_dict)
            len_test, total_correct = self.test_global()
            acc = total_correct / len_test
            self.data_frame.loc[epoch, "Accuracy"] = acc
            print(f"Global accuracy at round {epoch}: {acc}")
        if k is not None:
            if k == 0:
                if not os.path.exists(os.path.join(self.config["log_path"], "CrossValidation")):
                    os.makedirs(os.path.join(self.config["log_path"], "CrossValidation"))
                else:
                    import shutil
                    shutil.rmtree(os.path.join(self.config["log_path"], "CrossValidation"))
                    os.makedirs(os.path.join(self.config["log_path"], "CrossValidation"))
            fold_path = os.path.join(
                self.config["log_path"], "CrossValidation", f"fold_{k+1}")
            os.makedirs(fold_path)
            for client in clients:
                client.data_frame.to_csv(os.path.join(
                    fold_path, f"Client_{client.id}.csv"))
            return os.path.join(self.config["log_path"], "CrossValidation")
        else:
            import time

            np.random.seed(int(str(time.time())[:2]))
            appended = str(np.random.randint(1,100))
            np.random.seed(self.config["seed"])
            self.appended = appended
            os.makedirs(os.path.join(self.config["log_path"], f"log_{self.config['ds_name']}{appended}"))
            for client in clients:
                client.data_frame.to_csv(os.path.join(
                    self.config["log_path"], f"log_{self.config['ds_name']}{appended}"  , f"Client_{client.id}.csv"))
            return os.path.join(
                    self.config["log_path"], f"log_{self.config['ds_name']}{appended}")

    def test_global(self):
        self.global_model.eval()
        total_correct = 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x.to(self.device)
                y.to(self.device)
                logits = self.global_model(x)
                preds = np.argmax(logits, axis=1)
                correct = torch.eq(preds, y.flatten())
                correct = torch.sum(correct)
                total_correct += correct
        return self.len_test, total_correct

    # def test_global(self):
    #     self.global_model.eval()
    #     total_correct = 0
    #     with torch.no_grad():
    #         for x, y in self.test_loader:
    #             x.to(self.device)
    #             y.to(self.device)
    #             logits = self.global_model(x)
    #             preds = logits.argsort(axis=1)
    #             # Set all values that are not 0 or 1 to 0
    #             preds[preds == 0] = 1
    #             preds[(preds != 0) & (preds != 1)] = 0
    #             correct = torch.eq(preds, y)
    #             correct = torch.sum(correct) / 14
    #             total_correct += correct

    #     return self.len_test, total_correct


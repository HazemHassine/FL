from argparse import ArgumentParser
from loguru import logger
import argparse
import importlib
import medmnist
from medmnist import INFO
from utils import create_config, iid_partition, non_iid_partition
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The config file", required=False)
    args = parser.parse_args()
    
    if args.config is not None:
        config = importlib.import_module(args.config.replace("/", "."))
        config = config.config
        model = config["model"]
        print(model)
        try:
            from pprint import pprint
            config["model"] = None
            pprint(config)
        except ModuleNotFoundError:
            config["model"] = None
            print("{")
            for k, v in config:
                print(f"{k}: {v}")
            print("}")
        config["model"] = model
    else:
        config = create_config()
    print(model)


    info = INFO[config["ds_name"]]
    print(config["ds_name"])
    DataClass = getattr(medmnist, info["python_class"])
    if not os.path.exists(config["data_path"]):
        os.makedirs(config["data_path"])
    
    train_dataset = DataClass(root=config["data_path"], download=True, split="train")
    test_dataset = DataClass(root=config["data_path"], download=True, split="test")

    if not config["baseline"]:
        if config["iid"]:
            train_data_dict = iid_partition(train_dataset, config["num_clients"])
        else:
            train_data_dict = non_iid_partition(train_dataset, config["num_clients"])

        test_data_dict = iid_partition(test_dataset, config["num_clients"])
        algorithm = config["algorithm"].lower()
        print(algorithm)
        match algorithm:
            case "fedavg":
                from server import Server
                server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
                pass
            case "fedprox":
                from server import Server
                print("FEDPROX")
                server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
                pass
            case "fedreg":
                from server import Server
                server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
                pass
            case "fedbn":
                from fedbn_server import FedBNServer
                server = FedBNServer(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
                pass
            case _:
                raise NotImplementedError
        server.train()
    else:
        import torch
        from torch.optim import SGD
        from torch.utils.data import DataLoader
        import numpy as np
        from utils import CustomDataset
        model = config["model"](config["n_channels"], config["n_classes"])
        len_test_dataset = len(test_dataset)
        train_loader = DataLoader(CustomDataset(train_dataset, list(range(len(train_dataset)))), batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(CustomDataset(test_dataset, list(range(len_test_dataset))), batch_size=config["batch_size"], shuffle=False)
        optimizer = SGD(model.parameters(), lr=config["learning_rate"])
        train_loss = []
        loss_fn = config["criterion"]()
        model.train()
        model.to(config["device"])
        for e in tqdm(range(1, config["local_epochs"]+1)):
            temp_loss = []
            print(f"EPOCH {e}")
            for i, (x,y) in enumerate(train_loader):
                x = torch.tensor(x, requires_grad=True).to(config["device"])
                y = torch.tensor(y).to(config["device"])
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y.squeeze(-1))
                loss.backward()
                optimizer.step()
                temp_loss.append(loss.detach().item())
            temp_loss = np.mean(temp_loss)
            print(f"Loss at epoch {e}:{temp_loss}")
            train_loss.append(temp_loss)
            if config["evaluate"]:
                model.eval()
                with torch.no_grad():
                    for i, (x,y) in enumerate(test_loader):
                        x = torch.tensor(x).to(config["device"])
                        y = torch.tensor(y).to(config["device"])
                        logits = model(x)
                        preds = np.argmax(logits.detach().numpy(), axis=1)
                        correct = sum(np.equal(preds, y.flatten()))
                    print(f"Accuracy at epoch {e} is {correct/len_test_dataset}")
        for i, (x,y) in enumerate(test_loader):
            x = torch.tensor(x).to(config["device"])
            y = torch.tensor(y).to(config["device"])
            logits = model(x)
            preds = np.argmax(logits.detach().numpy(), axis=1)
            correct = sum(np.equal(preds, y.flatten()))

        print(f"Accuracy after {config['local_epochs']} of training is {correct/len_test_dataset}")
        # optimizer = SGD(self.model.parameters(), self.config["learning_rate"])
        # self.model.train()
        # train_loss = []
        # for l_epoch in range(self.config["local_epochs"]):
        #     for x, y in self.train_loader:
        #         x.to(self.device)
        #         y.to(self.device)
        #         optimizer.zero_grad()
        #         logits = self.model(x)
        #         loss = self.loss_fn(logits, y.squeeze(-1))
        #         loss.backward()
        #         optimizer.step()
        #         train_loss.append(loss.detach().item())
        # print(np.mean(train_loss))
        # return self.num_train_samples, self.model.state_dict()


if __name__=="__main__":
    main()
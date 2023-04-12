from argparse import ArgumentParser
from loguru import logger
import importlib
import medmnist
from medmnist import INFO
from utils import create_config, iid_partition, non_iid_partition, plots
import os
import warnings
import pandas as pd
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="The config file", required=False)
    args = parser.parse_args()

    if args.config is not None:
        config = importlib.import_module(args.config.replace("/", "."))
        config = config.config
        model = config["model"]
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

    info = INFO[config["ds_name"]]
    DataClass = getattr(medmnist, info["python_class"])
    if not os.path.exists(config["data_path"]):
        os.makedirs(config["data_path"])

    seed = config["seed"]
    import random
    import numpy as np
    import torch
    # set the seed for Python's built-in random module
    random.seed(seed)
    # set the seed for Numpy
    np.random.seed(seed)
    # set the seed for PyTorch
    torch.manual_seed(seed)

    train_dataset = DataClass(
        root=config["data_path"], download=True, split="train")
    test_dataset = DataClass(
        root=config["data_path"], download=True, split="test")
    
    try:
        CV = config["CV"]
        k = config["k"]
    except KeyError:
        CV = False

    if not config["baseline"]:
        if not CV:
            print("Training with no cross validation")
            if config["iid"]:
                train_data_dict = iid_partition(
                    train_dataset, config["num_clients"])
            else:
                train_data_dict = non_iid_partition(
                    train_dataset, config["num_clients"])

            test_data_dict = iid_partition(test_dataset, config["num_clients"])
            algorithm = config["algorithm"].lower()
            
            if algorithm == "fedavg":
                from server import Server
                server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
            elif algorithm == "fedprox":
                from server import Server
                server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
            elif algorithm == "fedreg":
                from server import Server
                server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
            elif algorithm == "fedbn":
                from fedbn_server import FedBNServer
                server = FedBNServer(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
            else:
                raise NotImplementedError

            path = server.train()
            plots(path)
        else:   
            from utils import split_list_k_folds
            print(f"Training using Cross validation {k}-folds")
            data = dict(np.load(os.path.join(config["data_path"], f"{config['ds_name']}.npz")))
            train_dataset = [(image, target) for image, target in zip(data["train_images"],data["train_labels"])]
            test_dataset = [(image, target) for image, target in zip(data["test_images"],data["test_labels"])]
            val_dataset = [(image, target) for image, target in zip(data["val_images"],data["val_labels"])]
            dataset_full = [*train_dataset, *test_dataset, *val_dataset]
            random.shuffle(dataset_full) # in place
            folds = split_list_k_folds(dataset_full, k)
            train = []
            for k in range(config["k"]):
                print(f"*****************FOLD {k+1}*****************")
                testing_data = folds[k]
                training_data = [fold for j, fold in enumerate(folds) if j != k]
                for split in training_data:
                    train = [*train, *split]
                
                if config["iid"]:
                    train_data_dict = iid_partition(train_dataset, config["num_clients"])
                else:
                    train_data_dict = non_iid_partition(train_dataset, config["num_clients"])
                test_data_dict = iid_partition(testing_data, config["num_clients"])
                algorithm = config["algorithm"].lower()
            
                if algorithm == "fedavg":
                    from server import Server
                    server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
                elif algorithm == "fedprox":
                    from server import Server
                    server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
                elif algorithm == "fedreg":
                    from server import Server
                    server = Server(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
                elif algorithm == "fedbn":
                    from fedbn_server import FedBNServer
                    server = FedBNServer(config, train_data_dict, test_data_dict, train_dataset, test_dataset)
                else:
                    raise NotImplementedError
            
                path = server.train(k)
                for client in server.clients:
                    client.data_frame = pd.DataFrame(columns=["Accuracy", "Loss"], index=list(range(1, config["global_epochs"]+1)))
            plots(path)
    else:
        import torch
        from torch.optim import SGD
        from torch.utils.data import DataLoader
        import numpy as np
        from utils import CustomDataset
        model = config["model"](config["n_channels"], config["n_classes"])
        len_test_dataset = len(test_dataset)
        train_loader = DataLoader(CustomDataset(train_dataset, list(
            range(len(train_dataset)))), batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(CustomDataset(test_dataset, list(
            range(len_test_dataset))), batch_size=config["batch_size"], shuffle=False)
        optimizer = SGD(model.parameters(), lr=config["learning_rate"])
        train_loss = []
        loss_fn = config["criterion"]()
        model.train()
        model.to(config["device"])
        for e in range(1, config["local_epochs"]+1):
            temp_loss = []
            print(f"EPOCH {e}")
            for i, (x, y) in enumerate(train_loader):
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
                    for i, (x, y) in enumerate(test_loader):
                        x = torch.tensor(x).to(config["device"])
                        y = torch.tensor(y).to(config["device"])
                        logits = model(x)
                        preds = np.argmax(logits.detach().numpy(), axis=1)
                        correct = sum(np.equal(preds, y.flatten()))
                    print(
                        f"Accuracy at epoch {e} is {correct/len_test_dataset}")
        for i, (x, y) in enumerate(test_loader):
            x = torch.tensor(x).to(config["device"])
            y = torch.tensor(y).to(config["device"])
            logits = model(x)
            preds = np.argmax(logits.detach().numpy(), axis=1)
            correct = sum(np.equal(preds, y.flatten()))

        print(
            f"Accuracy after {config['local_epochs']} of training is {correct/len_test_dataset}")


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from loguru import logger
import argparse
import importlib
import medmnist
from medmnist import INFO
from utils import create_config, iid_partition, non_iid_partition
import os
import pprint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The config file", required=False)
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
    print(config["ds_name"])
    DataClass = getattr(medmnist, info["python_class"])
    if not os.path.exists(config["data_path"]):
        os.makedirs(config["data_path"])
    
    train_dataset = DataClass(root=config["data_path"], download=True, split="train")
    test_dataset = DataClass(root=config["data_path"], download=True, split="test")

    if config["iid"]:
        train_data_dict = iid_partition(train_dataset, config["num_clients"])
    else:
        train_data_dict = non_iid_partition(train_dataset, config["num_clients"])

    test_data_dict = iid_partition(test_dataset, config["num_clients"])
    algorithm = config["algorithm"].lower()

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
    server.train()

if __name__=="__main__":
    main()
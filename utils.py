from torch.utils.data import Dataset
import torch
from medmnist import INFO
import os
import numpy as np
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, dataset, idxs, transform=None) -> None:
        self.dataset = dataset
        self.idxs = list(idxs)
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, index):
        if self.transform == None:
            return self.dataset[self.idxs[index]]
        else:
            x, y = self.dataset[self.idxs[index]]
            x = self.transform(x)
            return torch.tensor(x, requires_grad=True).permute((1, 2, 0)) if not type(x) == type(torch.tensor([1])) else x, torch.tensor(y)


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def create_config() -> dict:
    # TODO add fedreg's parameters
    config = {}
    while True:
        baseline = input("Use Federated Learning? [y/n]")
        if baseline.lower() not in ["n", "y"]:
            print("Not a valid choice [y/n]")
        else:
            baseline = False if baseline.lower() == "y" else True
            break

    while True:
        default = input('''
Use default folder configuration:
-data/ : for raw data storage (.npz files)
-logs/ : for the logs (.log files) [y/n]
''')
        if default.lower() not in ["n", "y"]:
            print("not a valid choice [y/n]")
        else:
            default = True if default.lower() == "y" else False
            break
    if not default:
        data_path = input('''Enter the data folder path:
''')
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        log_path = input('''Enter the log path:
''')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    else:
        if not os.path.exists("./data"):
            os.mkdir("data")
        if not os.path.exists("./logs"):
            os.mkdir("logs")
        data_path = "./data"
        log_path = "./logs"
    if not baseline:
        while True:
            algorithm = input('''
Choose between:
1- fedavg
2- fedprox        
3- fedbn        
4- fedreg
''')
            if algorithm.isdigit():
                if int(algorithm) not in [1, 2, 3, 4]:
                    print("Enter a valid number please (1/2/3/4)")
                else:
                    algos = {1: "fedavg", 2: "fedprox",
                             3: "fedbn", 4: "fedreg"}
                    algorithm = algos[int(algorithm)]
                    break
        while True:
            num_clients = input('''Number of total clients:
            ''')
            if num_clients.isdigit():
                num_clients = int(num_clients)
                break
            else:
                print("Enter a number")
        while True:
            participation_percent = input('''Participation percent (float in [0.0, 1.0]):
            ''')
            if isfloat(participation_percent):
                participation_percent = float(participation_percent)
                if participation_percent > 1 or participation_percent < 0:
                    print("Enter a float [0.0, 1.0]")
                    continue
                participation_percent = float(participation_percent)
                break
            else:
                print("Enter a float [0.0, 1.0]")
    while True:
        batch_size = input(f"Batch size")
        if batch_size.isdigit():
            if int(batch_size) <= 0:
                print("Enter a number greater than 0")
                continue
            batch_size = int(batch_size)
            break
        else:
            print("Enter a number")
    if baseline:
        while True:
            local_epochs = input("Number of epochs:")
            if local_epochs.isdigit():
                if int(local_epochs) <= 0:
                    print("Enter a number greater than 0")
                    continue
                local_epochs = int(local_epochs)
                break
            else:
                print("Enter a number")
    else:
        while True:
            local_epochs = input("Local Epochs:")
            if local_epochs.isdigit():
                if int(local_epochs) <= 0:
                    print("Enter a number greater than 0")
                    continue
                local_epochs = int(local_epochs)
                break
            else:
                print("Enter a number")
        while True:
            global_epochs = input("Number of communication rounds:")
            if global_epochs.isdigit():
                if int(global_epochs) <= 0:
                    print("Enter a number greater than 0")
                    continue
                global_epochs = int(global_epochs)
                break
            else:
                print("Enter a number")
    while True:
        learning_rate = input("Enter the learning rate:")
        if isfloat(learning_rate):
            if float(learning_rate) <= 0:
                print("Enter a float greater than 0")
                continue
            learning_rate = float(learning_rate)
            break
        else:
            print("Enter a number")
    while True:
        # TODO: import models etc..
        model_choice = input('''Choose a model:
1. CNN
2. resnet18
''')
        if model_choice.isdigit():
            if int(model_choice) not in [1, 2]:
                print("Enter either 1 or 2")
                continue
            match int(model_choice):
                case 1:
                    from models import CNN
                    model = CNN
                    pass
                case 2:
                    from models import ResNet18
                    model = ResNet18
                    pass
            break
        else:
            print("Enter either 1 or 2")
    if not baseline:
        while True:
            iid = input('''
Choose:
1- IID
2- non-IID
------
(Data heterogenity in terms of labels)
''')
            if iid.isdigit():
                if int(iid) not in [1, 2]:
                    print("Enter either 1 or 2")
                    continue
                iid = True if iid == 1 else False
                break
            else:
                print("Enter either 1 or 2")

        if algorithm in ["fedreg", "fedprox"]:
            while True:
                gamma = input("Enter the value of gamma/mu")
                if isfloat(gamma):
                    gamma = float(gamma)
                    break
                else:
                    print("Please enter the value of mu and gamma")
        else:
            gamma = None

    datasets = [*INFO.keys()][:-6]
    confirm = "r"
    while True:
        if confirm.lower() == "y":
            break
        for i, dataset in enumerate(datasets):
            print(f"{i} - {dataset}")
        dataset_choice = input("Enter the number corresponding to the dataset")
        if dataset_choice.isdigit():
            dataset_choice = int(dataset_choice)
            if int(dataset_choice) in list(range(len(datasets))):
                info = INFO[datasets[dataset_choice]]
                ds_name = datasets[dataset_choice]
                # info = INFO[list(INFO.keys())[dataset_choice]]
                num_channels = info["n_channels"]
                num_classes = len(info["label"])
                task = info["task"]
                print(f'''\t\t
**********DATASET DESCRIPTION**********
{info["description"]}
TASK: {task}
***************************************
''')
                print("Close the image please")
                import matplotlib.pyplot as plt
                img = plt.imread(f"visualization/{ds_name}.png")
                plt.imshow(img)
                plt.show()
                while True:
                    confirm = input("Do you confirm? [y/n] ")
                    if confirm.lower() == "y":
                        break
                    elif confirm.lower() == "n":
                        break
                    else:
                        print("Enter a valid option (y/n) ")
            else:
                print("Enter a valid choice", *list(range(len(datasets))))
        else:
            print("Enter a number please")
    while True:
        seed = input("Enter the randomness seed (pytroch/cuda/numpy/random)")
        if seed.isdigit():
            seed = int(seed)
            break
        else:
            print("Please enter a number")

    while True:
        CV = input("Use K-fold cross validation? [y/n]")
        if CV.lower() not in ["y", "n"]:
            print("Enter a valid choice [y/n]")
        else:
            CV = True if CV.lower() == "y" else False
            if not CV:
                k = None
            break
    if CV:
        while True:
            k = input(
            "Enter the number of folds (K), Enter: default (5)")
            if k == '':
                k = 5
                break
            if not k.isdigit():
                print("Please enter a digit")
                continue
            else:
                k = int(k)

    import torch
    if baseline:
        while True:
            evaluate = input("Evaluate while training? [y/n]")
            if evaluate.lower() not in ["y", "n"]:
                print("Please enter a valid choice [y/n]")
            else:
                evaluate = True if evaluate == "y" else False
                break

        config = {
            "baseline": baseline,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "ds_name": ds_name,
            "seed": seed,  # random seed
            # the model to be trained the p ps and pt are only relevant in the fedreg.
            "model": model,
            "n_classes": num_classes,
            "n_channels": num_channels,
            "task": task,
            "data_path": "./data" if default else data_path,
            "local_epochs": local_epochs,  # the number of epochs in local training stage
            "batch_size": batch_size,  # the batch size in local training stage
            "log_path": log_path,  # the path to save the log file
            # the preprocessing of train data, please refer to torchvision.transforms
            "train_transform": None,
            "test_transform": None,  # the preprocessing of test dasta
            "criterion":  nn.BCEWithLogitsLoss if task == "multi-label, binary-class" else nn.CrossEntropyLoss,
            "learning_rate": learning_rate,
            "evaluate": evaluate
        }
    else:
        config = {
            "baseline": baseline,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "ds_name": ds_name,
            "seed": seed,  # random seed
            # the model to be trained the p ps and pt are only relevant in the fedreg.
            "model": model,
            "algorithm": algorithm,  # FL optimizer, can be FedAvg, FedProx, FedCurv or SCAFFOLD
            "n_classes": num_classes,
            "n_channels": num_channels,
            "task": task,
            "data_path": "./data" if default else data_path,
            "num_clients": num_clients,
            "participation_percent": participation_percent,
            "global_epochs": global_epochs,
            "local_epochs": local_epochs,  # the number of epochs in local training stage
            "batch_size": batch_size,  # the batch size in local training stage
            "log_path": log_path,  # the path to save the log file
            # the preprocessing of train data, please refer to torchvision.transforms
            "train_transform": None,
            "test_transform": None,  # the preprocessing of test dasta
            "eval_train": True,  # whether to evaluate the model performance on the training data. Recommend to False when the training dataset is too large
            "gamma": gamma,  # the value of gamma when FedReg is used, the weight for the proximal term when FedProx is used, or the value of lambda when FedCurv is used
            "iid": iid,
            "criterion":  nn.BCEWithLogitsLoss if task == "multi-label, binary-class" else nn.CrossEntropyLoss,
            "learning_rate": learning_rate,
            "CV": CV,
            "k": k if CV else "No Cross validation"
        }
    while True:
        see = input("Do you want to see the config file [y/n]")
        if see.lower() not in ["y", "n"]:
            print("please write y for yes or n for no")
            continue
        elif see.lower() == "y":
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
            break
        break
    config["model"] = model

    return config


def non_iid_partition(dataset, num_clients):
    """
    non I.I.D parititioning of data over clients
    Sort the data by the digit label
    Divide the data into N shards of size S
    Each of the clients will get X shards

    params:
      - dataset (torch.utils.Dataset): Dataset containing the pathMNIST Images
      - num_clients (int): Number of Clients to split the data between
      - total_shards (int): Number of shards to partition the data in
      - shards_size (int): Size of each shard 
      - num_shards_per_client (int): Number of shards of size shards_size that each client receives

    returns:
      - Dictionary of image indexes for each client
    """

    if dataset[0][1].shape[0] != 1:
        from sklearn.cluster import KMeans
        # Create an array of arrays
        arr = np.array([np.array(target) for _, target in dataset])
        # Define the number of clusters
        n_clusters = num_clients
        # Create a KMeans instance
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        # Fit the KMeans model
        kmeans.fit(arr)
        # Get the labels for each array
        labels = kmeans.labels_
        # Get the indices of each data point in each cluster
        cluster_indices = {}
        for i in range(n_clusters):
            indices = np.where(labels == i)[0]
            cluster_indices[i] = indices
        return cluster_indices

    shards_size = 9
    total_shards = len(dataset) // shards_size
    num_shards_per_client = total_shards // num_clients
    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype='int64') for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    # get labels as a numpy array
    data_labels = np.array([np.array(target).flatten()
                           for _, target in dataset]).flatten()
    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1, :].argsort()]
    idxs = label_idxs[0, :]

    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(num_clients):
        rand_set = set(np.random.choice(
            shard_idxs, num_shards_per_client, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)

        for rand in rand_set:
            client_dict[i] = np.concatenate(
                (client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)
    return client_dict  # client dict has [idx: list(datapoint indices)


def iid_partition(dataset, clients):
    """
    I.I.D paritioning of data over clients
    Shuffle the data
    Split it between clients
    params:
      - dataset (torch.utils.Dataset): Dataset containing the PathMNIST Images 
      - clients (int): Number of Clients to split the data between
    returns:
      - Dictionary of image indexes for each client
    """
    num_items_per_client = int(len(dataset)/clients)
    client_dict = {}
    image_idxs = [i for i in range(len(dataset))]
    for i in range(clients):
        client_dict[i] = set(np.random.choice(
            image_idxs, num_items_per_client, replace=False))
        image_idxs = list(set(image_idxs) - client_dict[i])
    return client_dict


def FSGM(model, inp, label, iters, eta, criterion):
    '''
    the function implements the FGSM attack to generate adversarial examples 
    for the given model, inp, and label
    this function is usually called from the generate fake method from a Model class
    this usually the model argument will take the value of self, called from Model 
    '''

    inp.requires_grad = True
    minv, maxv = float(inp.min().detach().cpu().numpy()), float(
        inp.max().detach().cpu().numpy())
    for _ in range(iters):
        out = model(inp)

        loss = criterion(out, label.flatten()).mean()
        dp = torch.sign(torch.autograd.grad(loss, inp)[0])
        inp.data.add_(eta*dp.detach()).clamp(minv, maxv)
    return inp


def generate_fake(model, d, p_iters, ps_eta, pt_eta, task):
    x, y = d
    model.eval()
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    psuedo, perturb = x.detach(), x.detach()
    if psuedo.device != next(model.parameters()).device:
        psuedo = psuedo.to(next(model.parameters()).device)
        perturb = perturb.to(next(model.parameters()).device)
    psuedo = FSGM(model, psuedo, y, p_iters, ps_eta, criterion)
    perturb = FSGM(model, perturb, y, p_iters, pt_eta, criterion)
    psuedo_y, perturb_y = model(psuedo), model(perturb)
    return [psuedo, y, psuedo_y], [perturb, y, perturb_y]

def split_list_k_folds(dataset, k):
    import random
    random.shuffle(dataset)
    fold_size = len(dataset) // k
    folds = []
    for i in range(k):
        fold = dataset[i*fold_size:(i+1)*fold_size]
        folds.append(fold)
    return folds


def plots(path):
    all_csvs = {}
    if "CrossValidation" in os.listdir(path):
        csvs = []
        for fold in os.listdir(os.path.join(path, "CrossValidation")):
            for client in os.listdir(os.path.join(path, "CrossValidation", fold)):
                if client.endswith(".csv"):
                    csvs.append(pd.read_csv(client))
            for csv in csvs:
                plt.plot(csv.Loss)
                plt.show()
            
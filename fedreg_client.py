from typing import Tuple, OrderedDict
from torch.utils.data import DataLoader
from utils import CustomDataset
import torch
import numpy as np
from torch.optim import SGD
import copy
from utils import generate_fake
class FedRegClient():
    def __init__(self, id, config, train_dataset, test_dataset, data_idxs, test_idxs, gamma, ps_eta, pt_eta, p_iters) -> None:
        self.id = id
        self.train_loader = DataLoader(CustomDataset(train_dataset, data_idxs),batch_size=config["batch_size"], shuffle=True)
        self.test_dataset = test_dataset
        self.test_loader = DataLoader(CustomDataset(test_dataset, test_idxs), batch_size=32, shuffle=False)

        # TODO: add generate function to the model
        self.model = config["model"](config["n_channels"], config["n_classes"])
        self.device = config["device"]
        self.len_test = len(test_dataset)
        self.num_train_samples = len(data_idxs)
        self.optimizer = SGD(self.model.parameters(), lr=config["learning_rate"])
        self.loss_fn = config["criterion"]()
        self.gamma = gamma
        self.ps_eta = ps_eta
        self.pt_eta = pt_eta
        self.p_iters = p_iters
        psuedo_data = []
        perturb_data = []
        # def generate_fake(model, d, p_iters, ps_eta, pt_eta)
        
        # TODO: Generate pseudo and fake in the init using the below valuess
        
        # pseudo_data , perturbed_data = GENERATEFAKE(model, ps_eta, pt_eta, p_iters)
        #    def generate_fake(self, x, y):
        # return [psuedo, y, psuedo_y], [perturb, y, perturb_y]
# 
        # and maybe put it in a custom dataset? cuz the mapping of the indexes
        ### special for fedreg ###


    def train(self):
        gamma = self.gamma
        ps_eta = self.ps_eta
        pt_eta = self.pt_eta
        p_iters = self.p_iters
        beta = 0.5
        criterion = self.loss_fn()
        lr = self.config["learning_rate"]
        for epoch in range(self.config["local_epochs"]):
            median_model, old_model, penal_model = copy.deepcopy(self.global_model), copy.deepcopy(self.global_model), copy.deepcopy(self.global_model)
            median_parameters = list(median_model.parameters())
            old_parameters = list(old_model.parameters())
            penal_parameters = list(penal_model.parameters())
            median_model_opt = SGD(median_model.parameters(), lr=lr)
            old_model_opt = SGD(old_model.parameters(), lr=lr)
            penal_model_opt = SGD(penal_model.parameters(), lr=lr)
            parameters = list(self.global_model.parameters())
            psuedo_data = []
            perturb_data = []
            for d in self.train_lodaer:
                psuedo, perturb = generate_fake(self.global_model, p_iters, ps_eta, pt_eta)
                psuedo_data.append(psuedo)
                perturb_data.append(perturb)
            for i, (x, y) in self.train_loader:
                median_model_opt.zero_grad()
                old_model_opt.zero_grad()
                penal_model_opt.zero_grad()
                median_model.train()
                old_model.train()
                penal_model.train()
                pseudo_datapoint, perturbed_datapoint = pseudo_datapoint['''something'''], perturbed_datapoint['''something''']

                for params, median_params, old_params in zip(parameters, median_parameters, old_parameters):
                    median_params.data.copy_(gamma*params+(1-gamma)*old_params)

                mloss = criterion(median_model(x), y).mean()
                grad1 = torch.autograd.grad(mloss, median_parameters)

                for g1, p in zip(grad1, parameters):
                    p.data.add_(-lr*g1)

                for p, o, pp in zip(parameters, old_parameters, penal_parameters):
                    pp.data.copy_(p*beta+o*(1-beta))

                ploss = criterion(penal_model(pseudo_datapoint[0]), pseudo_datapoint[2]).mean()
                grad2 = torch.autograd.grad(ploss, penal_parameters)
                with torch.no_grad():
                    dtheta = [(p-o) for p, o in zip(parameters, old_parameters)]
                    s2 = sum([(g2*g2).sum() for g2 in grad2])
                    w_s = (sum([(g0*g2).sum() for g0, g2 in zip(dtheta, grad2)]))/s2.add(1e-30)
                    w_s = w_s.clamp(0.0, )

                pertub_ploss = criterion(penal_model(perturbed_datapoint[0]), perturbed_datapoint[1]).mean()
                grad3 = torch.autograd.grad(pertub_ploss, penal_parameters)
                s3 = sum([(g3*g3).sum() for g3 in grad3])
                w_p = (sum([((g0-w_s*g2)*g3).sum() for g0, g2, g3 in zip(dtheta, grad2, grad3)]))/s3.add(1e-30)
                w_p = w_p.clamp(0.0,)

                for g2, g3, p in zip(grad2, grad3, parameters):
                    p.data.add_(-w_s*g2-w_p*g3)
                median_model_opt.step()
                old_model_opt.step()
                penal_model_opt.step()

    # TODO: change the training function to include the pseduo and perturbed data losses and generation and everything
    def train(self):
        optimizer = SGD(self.model.parameters(), self.config["learning_rate"])
        gamma = self.gamma
        ps_eta = self.ps_eta
        pt_eta = self.pt_eta
        p_iters = self.p_iters
        
        self.model.train()
        train_loss = []
        for l_epoch in range(self.config["local_epochs"]):
            for x, y in self.train_loader:
                x.to(self.device)
                y.to(self.device)
                self.inner_optimizer.zero_grad()
                logits = self.model(x)
                loss = self.loss_fn(y, logits)
                loss.backward()
                self.inner_optimizer.step()
                train_loss.append(loss.cpu().numpy()[0])
        return train_loss

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
    
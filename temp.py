def step_func(model, data, fed):
    lr = model.learning_rate
    parameters = list(model.parameters())
    flop = model.flop
    gamma = fed.gamma
    add_mask = fed.add_mask
    beta = 0.5

    psuedo_data, perturb_data = [], []
    for d in data:
        x, y = d
        psuedo, perturb = fed.model.generate_fake(x, y)
        psuedo_data.append(psuedo)
        perturb_data.append(perturb)
    idx = 0
    median_model, old_model, penal_model = copy.deepcopy(fed.model), copy.deepcopy(fed.model), copy.deepcopy(fed.model)
    median_parameters = list(median_model.parameters())
    old_parameters = list(old_model.parameters())
    penal_parameters = list(penal_model.parameters())

    def temp(self):
        def func(d):
            nonlocal idx, add_mask, beta, flop, gamma, lr
            model.train()
            median_model.train()
            penal_model.train()
            model.zero_grad()
            median_model.zero_grad()
            penal_model.zero_grad()

            x, y = d
            pseudo_datapoint, perturbed_datapoint = psuedo_data[idx % len(psuedo_data)], perturb_data[idx % len(perturb_data)]
            idx += 1

            for params, median_params, old_params in zip(parameters, median_parameters, old_parameters):
                median_params.data.copy_(gamma*params+(1-gamma)*old_params)

            mloss = median_model.loss(median_model(x), y).mean()
            grad1 = torch.autograd.grad(mloss, median_parameters)

            for g1, p in zip(grad1, parameters):
                p.data.add_(-lr*g1)

            for p, o, pp in zip(parameters, old_parameters, penal_parameters):
                pp.data.copy_(p*beta+o*(1-beta))

            ploss = penal_model.loss(penal_model(pseudo_datapoint[0]), pseudo_datapoint[2]).mean()
            grad2 = torch.autograd.grad(ploss, penal_parameters)
            with torch.no_grad():
                dtheta = [(p-o) for p, o in zip(parameters, old_parameters)]
                s2 = sum([(g2*g2).sum() for g2 in grad2])
                w_s = (sum([(g0*g2).sum() for g0, g2 in zip(dtheta, grad2)]))/s2.add(1e-30)
                w_s = w_s.clamp(0.0, )

            pertub_ploss = penal_model.loss(penal_model(perturbed_datapoint[0]), perturbed_datapoint[1]).mean()
            grad3 = torch.autograd.grad(pertub_ploss, penal_parameters)
            s3 = sum([(g3*g3).sum() for g3 in grad3])
            w_p = (sum([((g0-w_s*g2)*g3).sum() for g0, g2, g3 in zip(dtheta, grad2, grad3)]))/s3.add(1e-30)
            w_p = w_p.clamp(0.0,)

            for g2, g3, p in zip(grad2, grad3, parameters):
                p.data.add_(-w_s*g2-w_p*g3)
            if add_mask:
                return flop*len(x)*4  # only consider the flop in NN
            else:
                return flop*len(x)*3
        return func
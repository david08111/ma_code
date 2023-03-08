import torch

class Optimizer_Wrapper():
    def __init__(self, net, optim_config):
        self.optim_type = optim_config.pop("optim_type")
        self.optim_config = optim_config

        self.optimizer = self.set_optimizer(net, self.optim_type, optim_config)

    def set_optimizer(self, net, optim_name, optim_config):
        if optim_name == "adam":
            return torch.optim.Adam(net.model.parameters(), **optim_config)
        elif optim_name == "adamw":
            return torch.optim.AdamW(net.model.parameters(), **optim_config)
        elif optim_name == "sgd":
            return torch.optim.SGD(net.parameters(), **optim_config)
        elif optim_name == "adagrad":
            return torch.optim.Adagrad(net.parameters(), **optim_config)
        elif optim_name == "rmsprop":
            return torch.optim.RMSprop(net.parameters(), **optim_config)
        elif optim_name == "amsgrad":
            return torch.optim.Adam(net.parameters(), amsgrad=True, **optim_config)
        elif optim_name == "adamax":
            return torch.optim.Adamax(net.parameters(), **optim_config)
        # if optim_name == "adam":
        #     return torch.optim.Adam(net.model.parameters(), lr=self.learning_rate, eps=self.eps, weight_decay=self.weight_decay)
        # elif optim_name == "sgd":
        #     return torch.optim.SGD(net.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        # elif optim_name == "adagrad":
        #     return torch.optim.Adagrad(net.parameters(), lr=self.learning_rate, lr_decay=self.learning_rate_decay, weight_decay=self.weight_decay, eps=self.eps)
        # elif optim_name == "rmsprop":
        #     return torch.optim.RMSprop(net.parameters(), lr=self.learning_rate, alpha=self.alpha, eps=self.eps, weight_decay=self.weight_decay, momentum=self.momentum)
        # elif optim_name == "amsgrad":
        #     return torch.optim.Adam(net.parameters(), lr=self.learning_rate, eps=self.eps, weight_decay=self.weight_decay, amsgrad=True)
        # elif optim_name == "adamax":
        #     return torch.optim.Adamax(net.parameters(), lr=self.learning_rate, eps=self.eps, weight_decay=self.weight_decay)

    def step(self):
        self.optimizer.step() # add closure method?

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def state_dict(self):
        return self.optimizer.state_dict()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_optimizer(self):
        return self.optimizer


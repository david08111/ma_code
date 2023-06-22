import torch

class Optimizer_Wrapper():
    def __init__(self, model_params, optim_config):
        self.optim_type = list(optim_config.keys())[0]
        self.optim_config = dict(optim_config[self.optim_type])
        # self.optim_type = optim_config.pop("optim_type")
        # self.optim_config = optim_config
        if "warmup" in self.optim_config:
            self.warmup = True
            self.warmup_config = self.optim_config.pop("warmup")
            self.warmup_config["lr_step_size"] = (self.warmup_config["end_lr"] - self.warmup_config["start_lr"]) / self.warmup_config["num_steps"]
            self.step = 0
            # self.step_fnct = self._step_warmup
        else:
            self.warmup = False
            # self.step_fnct = self.optimizer.step

        self.optimizer = self.set_optimizer(model_params, self.optim_type, self.optim_config)

    def set_optimizer(self, model_params, optim_name, optim_config):
        if optim_name == "adam":
            return torch.optim.Adam(model_params, **optim_config)
        elif optim_name == "adamw":
            return torch.optim.AdamW(model_params, **optim_config)
        elif optim_name == "sgd":
            return torch.optim.SGD(model_params, **optim_config)
        elif optim_name == "adagrad":
            return torch.optim.Adagrad(model_params, **optim_config)
        elif optim_name == "rmsprop":
            return torch.optim.RMSprop(model_params, **optim_config)
        elif optim_name == "amsgrad":
            return torch.optim.Adam(model_params, amsgrad=True, **optim_config)
        elif optim_name == "adamax":
            return torch.optim.Adamax(model_params, **optim_config)
        else:
            raise NameError(f"Optimizer {optim_name} not implemented!")

    def step(self):
        self.optimizer.step()

    # self.step_fnct()
    # def step(self):
    #     self.step_fnct()
    #
    # def _step_warmup(self):
    #     lr = self.warmup_config["start_lr"] + self.step * self.warmup_config["lr_step_size"]
    #     for p in self.optimizer.param_groups:
    #         p['lr'] = lr
    #     self.step += 1
    #
    #     if lr >= self.warmup_config["end_lr"]:
    #         self.step_fnct = self.optimizer.step
    #
    #     self.optimizer.step()

    def set_warmup_lr(self):
        if self.warmup:
            lr = self.warmup_config["start_lr"] + self.step * self.warmup_config["lr_step_size"]
            for p in self.optimizer.param_groups:
                p['lr'] = lr
            self.step += 1

            if lr >= self.warmup_config["end_lr"]:
                self.warmup = False

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


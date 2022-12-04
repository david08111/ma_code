import torch

class Scheduler_Wrapper():
    def __init__(self, scheduler_config, optimizer):
        self.scheduler_type = scheduler_config.pop("sched_type")
        self.scheduler_config = scheduler_config
        self.optimizer = optimizer
        self.scheduler = self.set_scheduler(self.scheduler_type, scheduler_config, self.optimizer)

    def set_scheduler(self, scheduler_type, scheduler_config, optimizer):
        if scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, mode="min", **scheduler_config)
        elif scheduler_type == "step_lr":
            return torch.optim.lr_scheduler.StepLR(optimizer.optimizer, **scheduler_config)
        elif scheduler_type == "exponential_lr":
            return torch.optim.lr_scheduler.ExponentialLR(optimizer.optimizer, **scheduler_config)
        elif scheduler_type == "cosine_annealing_lr":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, **scheduler_config)

    def step(self):
        self.optimizer.step()

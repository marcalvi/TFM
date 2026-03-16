from typing import *
import torch
import torch.nn as nn

def calc_reg_loss(model, l1: float, model_topo: str, sources: List[str]):

    if model_topo == "fcnn": # don't regularise FCNN
        reg_loss = 0
    elif model_topo == "mcat" and sources == ["omic"]:
        reg_loss = 0
    else:
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        reg_loss = float(l1) * l1_norm
    return reg_loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, mode='min'):
        """
        Constructor for early stopping.

        Parameters:
        - patience (int): How many epochs to wait before stopping once performance stops improving.
        - verbose (bool): If True, prints out a message for each validation metric improvement.
        - mode (str): One of ['min', 'max']. Minimize (e.g., loss) or maximize (e.g., accuracy) the metric.
        """
        assert mode in ['min', 'max'], "Mode must be 'min' or 'max'"
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        # Use plain Python floats and Python comparison operators to avoid dtype mixing issues
        if mode == 'min':
            self.best_metric = float('inf')
            self.operator = lambda a, b: a < b
        else:
            self.best_metric = float('-inf')
            self.operator = lambda a, b: a > b

        self.best_model_weights = None
        self.should_stop = False

    def step(self, metric, model):
        """
        Check the early stopping conditions.

        Parameters:
        - metric (float or tensor): The latest validation metric (loss, accuracy, etc.).
        - model (torch.nn.Module): The model being trained.

        Returns:
        - bool: True if early stopping conditions met, False otherwise.
        """
        # Normalize metric to a Python float (handles torch.Tensor, numpy types, Python numbers)
        if isinstance(metric, torch.Tensor):
            metric_val = float(metric.detach().cpu().item())
        else:
            metric_val = float(metric)

        if self.operator(metric_val, self.best_metric):
            if self.verbose:
                print(f"Validation metric improved from {self.best_metric:.4f} to {metric_val:.4f}. Saving model weights.")
            self.best_metric = metric_val
            self.counter = 0
            # copy state_dict to avoid accidental mutation
            self.best_model_weights = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation metric did not improve. Patience: {self.counter}/{self.patience}.")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def load_best_weights(self, model):
        """
        Load the best model weights.

        Parameters:
        - model (torch.nn.Module): The model to which the best weights should be loaded.
        """
        if self.verbose:
            print(f"Loading best model weights with validation metric value: {self.best_metric:.4f}")
        # best_model_weights were stored on CPU; move them back to model's device when loading
        device = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else torch.device('cpu')
        state_dict = {k: v.to(device) for k, v in self.best_model_weights.items()}
        model.load_state_dict(state_dict)
        return model
from torch import nn
from torch import optim
from typing import Iterable


def get_act(name, **kwargs):
    act_fn = getattr(nn, name)(**kwargs)
    return act_fn


def get_optim(name, params, **kwargs):
    optimizer = getattr(optim, name)(params, **kwargs)
    return optimizer


def get_parameters(modules: Iterable[nn.Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    """
    Parameter freezing based on:
    https://github.com/RajGhugare19/dreamerv2/blob/main/dreamerv2/utils/module.py
    """

    def __init__(self, modules: Iterable[nn.Module]):
        """
        Context manager to locally freeze gradients.
        
        In some cases with can speed up computation because gradients aren't calculated for these listed modules and save out on memory usage.
        
        Example:
            ```
            with FreezeParameters([module]):
                output_tensor = module(input_tensor)
            ```
            
        Args:
            modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

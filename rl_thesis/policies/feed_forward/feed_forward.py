from torch import nn
import torch
from rl_thesis.policies.torch_policy import TorchPolicy
from .config import Config
from rl_thesis.utils.hooks import HookFunction, BackwardHookFunction


class FeedForwardNN(TorchPolicy):
    def __init__(
        self,
        hidden_layer_sizes,
        input_dim,
        output_dim,
        act_fn,
        hook_fn : BackwardHookFunction = None
    ):
        """
        Args:
            hidden_layer_sizes: list of hidden layer dimensions
            input_dim: int of input dim.
            output_dim: int of output dim.
            act_fn: string specifying name of activation function to use (e.g. "ReLU", "ELU", etc.)
        """
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList([])
        layer_input_dim = input_dim
        self.hook_fn = hook_fn

        for layer_num, hidden_layer_size in enumerate(hidden_layer_sizes):
            layer = nn.Linear(layer_input_dim, hidden_layer_size)
            if hook_fn:
                layer.register_full_backward_hook(self.hook(f"{layer_num}"))
            layer_input_dim = hidden_layer_size
            self.layers.append(layer)
        self.output_layer = nn.Linear(layer_input_dim, output_dim)
        if hook_fn:
            self.output_layer.register_full_backward_hook(self.hook("output_layer"))
        self.act = getattr(nn, act_fn)()


    def hook(self, layer_name):
        return lambda m, i, o: self.hook_fn(layer_name, m, i, o)

    def forward(self, x: torch.tensor):
        for layer in self.layers:
            x = self.act(layer(x))
        out = self.output_layer(x)
        return out

    def on_episode_over(self):
        pass

    def toggle_optimization_step(self, _: bool):
        pass

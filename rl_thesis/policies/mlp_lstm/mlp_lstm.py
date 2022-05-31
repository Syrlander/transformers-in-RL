from torch import lstm, nn
import torch
from rl_thesis.policies.torch_policy import TorchPolicy
from .config import Config
from typing import List


class MLPLSTM(TorchPolicy):
    def __init__(
        self,
        input_size: int,
        layer_sizes_before_lstm: List[int],
        lstm_hidden_size: int,
        n_lstm_layers: int,
        layer_sizes_after_lstm: List[int],
        output_size: int,
        device: str,
    ) -> None:
        super(MLPLSTM, self).__init__()
        self.device = device

        # Setup layers before lstm
        self.layers_before = nn.ModuleList([])
        self.input_size = input_size
        layer_input_size = input_size
        for layer_size in layer_sizes_before_lstm:
            self.layers_before.append(nn.Linear(
                layer_input_size,
                layer_size,
            ))
            layer_input_size = layer_size

        # Setup lstm layers
        self.lstm_layer = nn.LSTM(
            input_size=layer_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
        )
        self.lstm_hidden_size = lstm_hidden_size
        layer_input_size = lstm_hidden_size

        # set up layers after lstm
        self.layers_after = nn.ModuleList([])
        for layer_size in layer_sizes_after_lstm:
            self.layers_before.append(nn.Linear(
                layer_input_size,
                layer_size,
            ))
            layer_input_size = layer_size
        self.output_layer = nn.Linear(layer_input_size, output_size)
        self.relu = nn.ReLU()
        
        # Setup hidden states in lstm
        self.reset_hidden_state()
        self.toggle_optimization_step(False)

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            batch_size, n_time_steps, input_size = x.shape
        elif len(x.shape) == 2:
            batch_size, input_size = x.shape
            n_time_steps = 1
        elif len(x.shape) == 1:
            input_size, = x.shape
            batch_size = 1
            n_time_steps = 1

        # TODO: Remove the 2 calls to .view, as dense layers can handle batch_size x n_timesteps inputs (only the last shape matters)
        x = x.view((batch_size * n_time_steps), input_size)

        for layer in self.layers_before:
            x = self.relu(layer(x))

        x = x.view((batch_size, n_time_steps, self.lstm_layer.input_size))
        # if we are in optimizing step we do not want to use previous hidden state, because different calls are independent, but this is not the case when evaluating
        if self.optimizing:
            lstm_out, _ = self.lstm_layer(x)
        else:
            lstm_out, next_hidden_state = self.lstm_layer(x, self.hidden_state)
            self.hidden_state = next_hidden_state

        out = lstm_out
        for layer in self.layers_after:
            out = self.relu(layer(lstm_out))

        out = self.output_layer(out)
        return out

    def reset_hidden_state(self):
        """
            Should be called when starting training on elements that are not in the same sequence as the previous elements, eg. when starting training on a new episode
        """
        self.hidden_state = (
            torch.zeros((1, 1, self.lstm_hidden_size)).to(self.device),
            torch.zeros((1, 1, self.lstm_hidden_size)).to(self.device),
        )

    def predict(self, x: torch.Tensor, deterministic=True):
        if not type(x) == torch.Tensor:
            x = torch.Tensor(x).to(self.device)
        with torch.no_grad():
            y = self.forward(x)
        if deterministic:
            return y.argmax(dim=-1).detach().cpu().numpy().reshape(-1), None
        else:
            distribution = torch.distributions.Categorical(y)
            return distribution.sample(), None

    def on_episode_over(self):
        self.reset_hidden_state()

    def toggle_optimization_step(self, optimizing: bool):
        self.optimizing = optimizing

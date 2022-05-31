from torch import lstm, nn
import torch
from rl_thesis.policies.torch_policy import TorchPolicy
from .config import Config


class RecurrentCNN(TorchPolicy):
    """
        Implementation of model from https://arxiv.org/pdf/1507.06527.pdf along with default parameters from that paper.
    """

    def __init__(self, filter_counts, strides, conv_sizes, lstm_hidden_dim,
                 output_dim, n_input_channels, lstm_input_dim, batch_size,
                 device):
        """
            batch_size should be the batch size used during optimization, during experience collection it is allowed to vary
        """
        super(RecurrentCNN, self).__init__()
        self.conv_layers = nn.ModuleList([])
        input_channel_count = n_input_channels
        for n_filters, stride, conv_size in zip(filter_counts, strides,
                                                conv_sizes):
            layer = nn.Conv2d(in_channels=input_channel_count,
                              out_channels=n_filters,
                              kernel_size=conv_size,
                              stride=stride)
            input_channel_count = n_filters
            self.conv_layers.append(layer)

        self.lstm = nn.LSTM(input_size=lstm_input_dim,
                            hidden_size=lstm_hidden_dim,
                            batch_first=True)
        self.linear = nn.Linear(lstm_hidden_dim, output_dim)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.batch_size = batch_size
        self.optimizing = False
        self.device = device
        self.reset_hidden_state()

    def forward(self, x: torch.tensor):
        """
            x is input to model shaped as (batch_size, n_time_steps, n_channels, h, w), if dim(x) < 5 batch_size and/or n_time_steps will be assumed to be 1
        """
        print(x.shape)
        if len(x.shape) == 5:
            batch_size, n_time_steps, n_channels, h, w = x.shape
        elif len(x.shape) == 4:
            batch_size, n_channels, h, w = x.shape
            n_time_steps = 1
        elif len(x.shape) == 3:
            n_channels, h, w = x.shape
            batch_size = 1
            n_time_steps = 1
        # reshape x so we can convolve all frames as one batch
        x = x.view((batch_size * n_time_steps, n_channels, h, w))
        for conv in self.conv_layers:
            x = self.relu(conv(x))
        # reshape x to shape expected by LSTM
        x = x.view((batch_size, n_time_steps, self.lstm_input_dim))

        # if we are in optimizing step we do not want to use previous hidden state, because different calls are independent, but this is not the case when evaluating
        if self.optimizing:
            lstm_out, _ = self.lstm(x)
        else:
            lstm_out, next_hidden_state = self.lstm(x, self.hidden_state)
            self.hidden_state = next_hidden_state
        # print(f"lstm shape: {lstm_out.shape}")
        out = self.linear(lstm_out)
        # print(f"result shape: {out.shape}")
        return out

    def predict(self, x : torch.Tensor, deterministic=True):
        self.toggle_optimization_step(False)
        if not type(x) == torch.Tensor:
            x = torch.Tensor(x)
        y = self.forward(x)
        if deterministic:
            return y.argmax(dim=-1).detach().cpu(), None
        else:
            distribution = torch.distributions.Categorical(y)
            return distribution.sample(), None

    def reset_hidden_state(self):
        """
            Should be called when starting training on elements that are not in the same sequence as the previous elements, eg. when starting training on a new episode
        """
        self.hidden_state = (torch.zeros(
            (1, self.batch_size, self.lstm_hidden_dim)).to(self.device),
                             torch.zeros(
                                 (1, self.batch_size,
                                  self.lstm_hidden_dim)).to(self.device))

    def on_episode_over(self):
        self.reset_hidden_state()

    def toggle_optimization_step(self, optimizing: bool):
        self.optimizing = optimizing

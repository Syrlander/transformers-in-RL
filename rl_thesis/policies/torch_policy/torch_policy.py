from torch import nn


class TorchPolicy(nn.Module):
    def __init__(self) -> None:
        super(TorchPolicy, self).__init__()

    def on_episode_over(self):
        """
            Called from torch_DQN model when an episode ends. Could for example be used to reset hidden states. Will be called both during evaluation and while collecting experience
        """
        pass

    def toggle_optimization_step(self, optimizing: bool):
        """
            If model requires specific settings set to do optimization (eg. during Q-learning) use this to set those settings
        """
        pass

    def forward(self, x):
        pass

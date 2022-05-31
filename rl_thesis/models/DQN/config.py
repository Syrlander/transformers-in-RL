from rl_thesis.models import BaseModelConfig


class Config(BaseModelConfig):
    """
    Default DQN config, hyperparameters based on:
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml#L35
    """
    batch_size=128
    buffer_size=10000
    learning_starts=1000
    gamma=0.98
    target_update_interval=600
    train_freq=16
    gradient_steps=8
    exploration_fraction=0.2
    exploration_final_eps=0.07
    learning_rate=4e-3

    def __init__(self):
        super().__init__()
        [
            setattr(self, k, v) 
            for k, v in vars(Config).items() 
               if not k.startswith("_")
        ]
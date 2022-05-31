from rl_thesis.models import BaseModelConfig


class Config(BaseModelConfig):
    """
    Default PPO config, hyperparameters based on:
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml#L45
    """
    normalize = True
    n_envs = 16
    n_steps=16
    gae_lambda=0.98
    gamma=0.99
    n_epochs=4
    ent_coef=0.0

    learning_rate = 0.0003
    batch_size = 64

    def __init__(self):
        super().__init__()
        [
            setattr(self, k, v) 
            for k, v in vars(Config).items() 
               if not k.startswith("_")
        ]

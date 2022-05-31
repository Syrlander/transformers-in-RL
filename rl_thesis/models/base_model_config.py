from pathlib import Path
from rl_thesis.config import BaseConfig

class BaseModelConfig(BaseConfig):
    """
    eval_post_train:
        Whether or not to evaluate model after training
    verbose:
        Whether or not to print logs
    """
    device : str = "cpu"
    verbose : bool = 0
    seed : int = 847592
    only_save_best : bool = True
    checkpoint_freq : int = 0 # How often to save model no matter if it was better than previous models or not, set to 0 to disable checkpoints

    normalize : bool = False
    n_envs : int = 1
    n_eval_episodes : int = 5
    eval_freq : int = 10000

    total_timesteps : int = 1e5
    
    models_dir : str = Path("models")
    log_dir : str = Path("train_reward_logs")

    def __init__(self):
        [
            setattr(self, k, v) 
            for k, v in vars(BaseModelConfig).items() 
                if not k.startswith("_")
        ]

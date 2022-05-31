from rl_thesis.models import BaseModelConfig
from typing import Tuple


class Config(BaseModelConfig):
    # Model
    state_rep : str = "gtrxl"
    n_latent_var : int = 32
    n_layer : int = 4
    n_head : int = 8
    dropout : float = 0.0
    mem_len : int = 10
    encoder : str = "ResNet"
    mlp_encoder_dim : int = None
    mlp_encoder_layers : int = None

    # Training
    log_dir: str = "GTrXL_logs" # Location of debugging metrics logs
    max_episodes : int = 1000           # Number of episodes to run for training
    max_episode_timesteps : int = 500   
    #update_timestep : int = 100         # Number of timesteps between each agent update
    update_episode : int = 128
    capacity : int = 1000
    batch_size : int = 128
    unroll_length : int = 95
    init_alpha : float = 5.0
    eps_alpha : float = 0.01
    init_eta : float = 1.0
    eps_eta : float = 0.1
    lr : float = 0.001 # learning rate is fixed to 1e-4 for all experiments - see VMPO paper, page 4, last paragraph of section 4.
    betas : Tuple[float] = (0.99, 0.999)
    gamma : float = 0.99 # discount
    K_epochs : int = 4 # == T_target, since the policy_old is first set after K_epoch iterations in VMPO.update

    def __init__(self):
        super().__init__()
        [
            setattr(self, k, v) for k, v in vars(Config).items()
            if not k.startswith("_") and not callable(v)
        ]

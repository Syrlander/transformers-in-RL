from rl_thesis.models import BaseModelConfig
from collections.abc import Mapping
from pathlib import Path, PosixPath
import json
from copy import deepcopy


class KwargsConfig(Mapping):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key : str):
        return getattr(self, key)

    def __iter__(self):
        for k in vars(self).keys():
            yield k

    def __len__(self):
        return len(vars(self))

    def get(self, key: str, default = None):
        try:
            return self.__getitem__(key)
        except AttributeError as e:
            if default:
                return default
            else:
                raise e


class Config(BaseModelConfig):
    log_dir = Path("dreamer_logs")
    prefill: int = 10000
    render_size = (64, 64)
    action_repeat = 1
    log_every = 1e4
    train_every = 5
    train_steps = 1
    expl_until = 0
    pretrain = 1
    debug = False # If True, model will not be saved and gradients will be logged
    replay = KwargsConfig(
        capacity=2e6,
        prioritized_replay_factor = 0,
        priority_min_reward = 0,
        favor_episode_ends = True
    )
    dataset = KwargsConfig(
        batch_size=16,
        seq_len=50,
        replace=False
    )
    log_keys_video = ["image"]
    log_keys_sum = '^$'
    log_keys_mean = '^$'
    log_keys_max = '^$'
    
    agent = KwargsConfig(
        expl_behavior = "greedy",
        expl_noise = 0,
        eval_noise = 0,
        eval_state_mean = False,
        world_model = KwargsConfig(
            grad_heads = ["decoder", "reward", "discount"],
            pred_discount = True,
            clip_rewards = "tanh",
            discount = 0.99,
            recurrence_model = "rssm", # rssm or tssm
            use_autocast = True,
            rssm = KwargsConfig(
                ensemble=1,
                hidden_state_dim=1024,
                deterministic_state_dim=1024,
                model_state_size=400,
                stoch_size=32,
                discrete=32,
                act_fn="ELU",
                std_act="sigmoid2", 
                min_std=0.1,
                action_space_size=3,
                transformer_config = KwargsConfig(
                    nhead=8, 
                    num_encoder_layers=6, 
                    num_decoder_layers=6, 
                    dim_feedforward=2048, 
                    dropout=0.1, 
                    activation="relu", 
                    layer_norm_eps=1e-05, 
                )
            ),
            encoder = KwargsConfig(
                mlp_keys = ".*",
                cnn_keys = ".*", 
                act = "ELU", 
                cnn_depth = 48, 
                cnn_kernels = [4, 4, 4, 4], 
                cnn_strides=2,
                mlp_layers = [400, 400, 400],
                mlp_output_dim = 400,
            ),
            decoder = KwargsConfig(
                mlp_keys = '.*',
                cnn_keys = '.*',
                act = "ELU",
                cnn_depth = 48,
                cnn_kernels = [5, 5, 6, 6], 
                mlp_layers = [400, 400, 400],
                mlp_output_dim = 400,
                model_state_size = 2048
            ),
            reward_head = KwargsConfig(
                hidden_layer_sizes = [400, 400, 400, 400], 
                input_dim = 2048,
                output_dim = [],
                act_fn = "ELU", 
                dist = "mse"
            ),
            discount_head = KwargsConfig(
                hidden_layer_sizes = [400, 400, 400, 400], 
                input_dim = 2048,
                output_dim = [],
                act_fn = "ELU",
                dist = "binary"
            ),
            loss_scales = KwargsConfig(
                kl = 1.0,
                reward = 1.0,
                discount = 1.0,
                proprio = 1.0
            ),
            kl = KwargsConfig(
                free = 0.0, 
                forward = False, 
                balance = 0.8, 
                free_avg = True
            ),
            model_opt = KwargsConfig(
                name="AdamW",
                kw_args = KwargsConfig(
                    lr = 1e-4, 
                    eps = 1e-5, 
                    weight_decay = 1e-6
                ),
                clip = 100,
            ),
        ),
        actor_critic = KwargsConfig(
            actor = KwargsConfig(
                hidden_layer_sizes = [400, 400, 400, 400],
                input_dim = 2048,
                act_fn = "ELU",
                dist = "auto",
                min_std = 0.1
            ),
            critic = KwargsConfig(
                hidden_layer_sizes=[400, 400, 400, 400],
                input_dim=2048,
                output_dim=[],
                act_fn = "ELU",
                dist = "mse",
            ),
            actor_opt = KwargsConfig(
                clip = 100,
                kw_args = KwargsConfig(
                    lr = 8e-5,
                    eps = 1e-5,
                    weight_decay = 1e-6
                )
            ),
            critic_opt = KwargsConfig(
                clip = 100,
                kw_args = KwargsConfig(
                    lr = 8e-5,
                    eps = 1e-5,
                    weight_decay = 1e-6
                )
            ),
            reward_normalizer = KwargsConfig(
                shape = (),
                momentum = 1.0,
                scale=1.0,
                eps=1e-8
            ),
            discount_lambda = 0.95,
            imag_horizon = 15,
            actor_grad = "auto",
            actor_grad_mix = 0.1,
            actor_ent = 2e-3,
            slow_target = True,
            slow_target_update = 100,
            slow_target_fraction = 1,
            slow_baseline = True,
            use_autocast = True
        )
    )

    def __init__(self):
        super().__init__()
        [
            setattr(self, k, v) for k, v in vars(Config).items()
            if not k.startswith("_") and not callable(v)
        ]



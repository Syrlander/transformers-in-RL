from typing import Union
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymStepReturn
import importlib
import numpy as np
from rl_thesis.config_parsing import overwrite_default_values
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


def setup_environment(env_name, n_envs=1, normalize=False, is_dreamer_env=False, monitor_dir=None, config_file=None):   
    # environments has the format `{env_name}-{version}` our custom envs use module_name as env_name
    env_kwargs = {}
    
    module_name = env_name.split('-')[0]
    try:
        # Attempt to load custom environment and pass env. config to it
        env_module = importlib.import_module(f"rl_thesis.environments.{module_name}")
        conf = env_module.Config()
        if config_file:
            overwrite_default_values(config_file, conf)

        env_kwargs["config"] = conf
    except ModuleNotFoundError:
        print(f"Did not find environment {env_name} in rl_thesis, assuming it is a gym env")
        pass

    if is_dreamer_env:
        #env = make_vec_env(env_name, n_envs=n_envs, monitor_dir=monitor_dir, wrapper_class=DreamerGymWrapper)
        env = gym.make(env_name, **env_kwargs)
        assert normalize == False
    else:
        env = make_vec_env(env_name, n_envs=n_envs, monitor_dir=monitor_dir, env_kwargs=env_kwargs)

    if normalize:
        print("Normalizing environment")
        env = VecNormalize(env)

    return env

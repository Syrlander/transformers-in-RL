from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from typing import Tuple, Type, Union, Any, Dict, Optional
from stable_baselines3.common.type_aliases import *
import numpy as np


class RandomPolicy(BasePolicy):
    def __init__(self, *args, squash_output: bool = False, **kwargs):
        super().__init__(*args, squash_output=squash_output, **kwargs)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return np.array([self.action_space.sample()])

    def predict(self, observation: Union[np.ndarray, Dict[str, np.ndarray]], state: Optional[Tuple[np.ndarray, ...]] = None, episode_start: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        action = self.action_space.sample()
        #print(f"action: {action}")
        return np.array([action]), state

    def forward(self, *args, **kwargs):
        pass

    def __call__(self, observation: Union[np.ndarray, Dict[str, np.ndarray]], state: Optional[Tuple[np.ndarray, ...]] = None, episode_start: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.predict(observation, state, episode_start, deterministic)

class Random(BaseAlgorithm):
    def __init__(self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str, None],
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        _init_setup_model=False
    ):
        #observation_space = env.observation_space
        #action_space = env.action_space
        
        #print(f"random policy before: {self.policy}")
        super(Random, self).__init__(None, env, None, 0, None, tensorboard_log, verbose, "cpu", support_multi_env, create_eval_env, monitor_wrapper, seed, use_sde, sde_sample_freq, supported_action_spaces)
        #print(f"random policy after: {self.policy}")
        self.policy = RandomPolicy(env.observation_space, env.action_space)

    def _setup_model(self):
        pass

    def learn(self):
        pass

    def eval(self):
        pass

from gym.envs.classic_control import MountainCarEnv
import numpy as np
from gym import spaces
import gym
from typing import List


class MountainCarNoVelocity(gym.Env):

    def __init__(self, config):
        super(MountainCarNoVelocity, self).__init__()
        self.observation_space = spaces.Box(low=-1.2, high=0.6, shape=(1, ))
        self.env = gym.make("MountainCar-v0")
        self.action_space = self.env.action_space

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        pos = np.array([state[0]])
        return pos, reward, done, info

    def reset(self):
        state = self.env.reset()
        position = np.array([state[0]])
        return position

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self) -> None:
        return self.env.close()

    def seed(self, seed: int = ...) -> List[int]:
        return self.env.seed(seed)

from gym.envs.classic_control import MountainCarEnv
import numpy as np
from gym import spaces
import gym
from typing import List
import matplotlib.pyplot as plt
import cv2


class MountainCarScreenInput(gym.Env):

    def __init__(self, config):
        super(MountainCarScreenInput, self).__init__()
        self.greyscale = config.greyscale
        observation_shape = (3, 84, 84) if not self.greyscale else (1, 84, 84)
        self.observation_space = spaces.Box(low=0, high=1, shape=observation_shape, dtype=np.float)
        self.env = gym.make("MountainCar-v0")
        self.action_space = self.env.action_space

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        self.img.set_data(self.render(mode="rgb_array")) # (400, 600, 3)
        self.img = cv2.resize(self.img, (84, 84)) # Resize image to (84, 84)
        self.img = np.transpose(self.img, axes=(2, 0, 1)) # (3, 84, 84)
        self.img = self.img / 255

        if self.greyscale:
            self.img = np.mean(self.img, axis=0)

        return self.img, reward, done, info

    def reset(self):
        state = self.env.reset()
        position = np.array([state[0]])
        self.img = plt.imshow(self.render(mode="rgb_array"))
        return position

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self) -> None:
        return self.env.close()

    def seed(self, seed: int = ...) -> List[int]:
        return self.env.seed(seed)

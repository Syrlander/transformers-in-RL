from cgitb import grey
from turtle import Screen
import numpy as np
from module import Memory, VMPO, PPO
from utils import set_up_hyperparams
import gym
from tqdm import tqdm
from PIL import Image, ImageOps


class ScreenObservationWrapper:

    def __init__(self, env, normalize=False, greyscale=False, resize=64):
        self._env = env
        self._normalize = normalize
        self._greyscale = greyscale
        self._resize = resize != None
        self._size = resize

        if greyscale:
            if self._resize:
                self._obs_shape = (self._size, self._size)
            else:
                self._obs_shape = (400, 600)
        else:
            if self._resize:
                self._obs_shape = (self._size, self._size, 3)
            else:
                self._obs_shape = (400, 600, 3)

    @property
    def observation_space(self):
        return self._obs_shape

    def step(self, action):
        _obs, reward, done, info = self.env.step(action)
        img = self._env.render(mode="rgb_array", close=True)
        return self.__apply_transforms(img, reward, done, info)

    def reset(self):
        self._env.reset()
        img = self._env.render(mode="rgb_array")
        return self.__apply_transforms(img)

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()

    def seed(self, seed):
        return self._env.seed(seed)

    def __apply_transforms(self, img):
        img = Image.fromarray(img, mode="RGB")

        if self._greyscale:
            img = self.__greyscale(img)

        if self._resize:
            img = self.__resize(img)

        img = np.array(img) # To image here for possible normalize

        if self._normalize:
            img = self.__normalize(img)

        return img

    def __greyscale(self, img):
        return ImageOps.grayscale(img)

    def __resize(self, img):
        img = img.resize(self._obs_shape[:2], Image.NEAREST)
        return img

    def __normalize(self, img):
        return img.astype(np.float) / 255.0


def main():
    H, logprint = set_up_hyperparams()

    # H.img_size = 64
    H.device = "cuda:" + H.gpu if H.gpu is not None else "cpu"

    memory = Memory()
    if H.model == "vmpo":
        agent = VMPO(H)
    elif H.model == "ppo":
        agent = PPO(H)

    env = gym.make(H.env_name)
    # env = ScreenObservationWrapper(gym.make(H.env_name))
    H.observation_dim = env.observation_space

    episode_returns = [ ]
    curr_return = 0
    update_timestep = 0

    for i_episode in tqdm(range(1, H.max_episodes + 1)):
        obs = env.reset()

        # For testing of observations after applying image transformations
        # img = Image.fromarray(obs, mode="L")
        # img = Image.fromarray(obs, mode="RGB")
        # img.save("obs_test.png")

        for t in range(H.max_timesteps):
            action = agent.policy_old.act(t, obs, memory)
            obs, reward, done, _info = env.step(action)

            update_timestep += 1

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            curr_return += reward

            if done:
                episode_returns.append(curr_return)
                curr_return = 0
                break

        if update_timestep > H.update_timestep:
            agent.update(memory)
            memory.clear_memory()
            update_timestep = 0

    env.close()

    episode_returns = np.array(episode_returns)
    print(f"All episode returns: {episode_returns}", end="\n\n")
    print("Episode returns:")
    print(f"\tAverage: {np.mean(episode_returns)}")
    print(f"\tMin: {np.min(episode_returns)}")
    print(f"\tMax: {np.max(episode_returns)}")


if __name__ == "__main__":
    main()

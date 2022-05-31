from rl_thesis.environments.utils import setup_environment
import numpy as np
from tqdm import tqdm
from rl_thesis.models.Dreamer import Model as Dreamer
from rl_thesis.dreamer.driver import Driver
from rl_thesis.environments import DreamerGymWrapper, OneHotAction
from rl_thesis.models.GTrXL import VMPOMemory
from rl_thesis.models.GTrXL import Model as GTrXL
import torch

def evaluate(model_class, model_filepath, env_name, num_timesteps, env_normalize=False, env_config=None, device="cpu"):
    """
    Note: env_normalize is not stored in env_config as it is model dependent
    """
    is_dreamer = model_class is Dreamer
    is_gtrxl = model_class is GTrXL
    env = setup_environment(env_name, normalize=env_normalize, config_file=env_config, is_dreamer_env=is_dreamer)
    print(f"Evaluating using device: {device}")
    model = model_class.load(model_filepath, env=env, device=device)
    model.eval()

    eps_returns, eps_lengths = [ ], [ ]

    if is_dreamer:
        env = OneHotAction(DreamerGymWrapper(env, obs_key="ram"))
        driver = Driver([env])
        # eval_step = lambda transition, worker: env.step()
        # driver.on_step(render_step)
        # driver.on_reset(render_step)

        def print_episode_stats(*args):
            episode = args[0]
            eps_returns.append(np.sum(episode['reward']))
            eps_lengths.append(len(episode['reward']))

        driver.on_episode(print_episode_stats)
        driver.on_episode(lambda _: driver.reset())

        def policy(*args):
            with torch.no_grad():
                return model.policy(*args, mode="eval")

        driver(policy, steps=num_timesteps)

        env.close()
    elif is_gtrxl:
        model.policy_old.device = device
        model.policy_old.shared_layer.device = device

        memory = VMPOMemory(num_timesteps)

        obs = env.reset()
        eps_return, eps_length = 0, 0

        for _ in tqdm(range(num_timesteps), desc="Running timesteps"):
            action = model.policy_old.act(eps_length, obs, memory)
            obs, reward, done, _info = env.envs[0].step(action)

            memory.push(obs, action, None, reward, eps_length, done)

            eps_return += reward
            eps_length += 1

            if done:
                memory.clear()
                eps_returns.append(eps_return)
                eps_lengths.append(eps_length)

                obs = env.reset()
                eps_return, eps_length = 0, 0
        env.close()
    else:
        env = env.envs[0]

        obs = env.reset()
        eps_return, eps_length = 0, 0
        
        for _ in tqdm(range(num_timesteps), desc="Running timesteps"):
            action, _state = model.predict(obs)
            try:
                if len(action) == 1:
                        action = action[0]
            except TypeError:
                pass
            obs, reward, done, _info = env.envs[0].step(action)

            if env_normalize:
                reward = env.get_original_reward()

            eps_return += reward
            eps_length += 1

            if done:
                eps_returns.append(eps_return)
                eps_lengths.append(eps_length)

                obs = env.reset()
                eps_return, eps_length = 0, 0

        assert len(eps_returns) == len(eps_lengths)

    print(f"\tNumber of episodes: {len(eps_lengths)}")
    print("")

    try:
        print("Length stats.:")
        print(f"\tAvg. episode length: {np.mean(eps_lengths)}")
        print(f"\tStd. episode length: {np.std(eps_lengths)}")
        print(f"\tMax. episode length: {np.max(eps_lengths)}")
        print(f"\tMin. episode length: {np.min(eps_lengths)}")
        print(f"\tMedian episode length: {np.median(eps_lengths)}")
        print("")

        print("Return stats.:")
        print(f"\tAvg. episode return: {np.mean(eps_returns)}")
        print(f"\tStd. episode return: {np.std(eps_returns)}")
        print(f"\tMax. episode return: {np.max(eps_returns)}")
        print(f"\tMin. episode return: {np.min(eps_returns)}")
        print(f"\tMedian episode return: {np.median(eps_returns)}")
    except ValueError:
        print("Stats. unavailable as no episodes complete. Increase num_timesteps!")
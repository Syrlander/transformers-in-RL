from rl_thesis.environments.utils import setup_environment
from rl_thesis.models import Dreamer
from rl_thesis.environments import DreamerGymWrapper, OneHotAction
from rl_thesis.dreamer.driver import Driver
import torch
import numpy as np


def render(model_class,
           model_filepath,
           env_name,
           num_timesteps,
           render_mode="human",
           env_normalize=False,
           env_config=None,
           device="cpu",
           **kwargs):
    """
    Note: env_normalize is not stored in env_config as it is model dependent
    """
    is_dreamer = model_class == Dreamer.model.DreamerModel

    env = setup_environment(env_name,
                            normalize=env_normalize,
                            config_file=env_config,
                            is_dreamer_env=is_dreamer)

    model = model_class.load(model_filepath, env=env, device=device)
    try:
        model.eval()
    except:
        pass

    if is_dreamer:
        # TODO: Maybe move the Dreamer env. wrapping into setup_environment and make the obs_key an environment config option
        env = OneHotAction(DreamerGymWrapper(env, obs_key="ram"))
        driver = Driver([env])
        render_step = lambda transition, worker: env.render(mode=render_mode)
        driver.on_step(render_step)
        driver.on_reset(render_step)
    
        def print_step_stats(transition, worker):
            print(f"ram: {transition['ram']}")
            print(f"reward: {transition['reward']}")
            print("")

        def print_episode_stats(*args):
            episode = args[0]
            print(f"Episode return: {np.sum(episode['reward'])}")

        driver.on_step(print_step_stats)
        driver.on_episode(print_episode_stats)

        def policy(*args):
            with torch.no_grad():
                return model.policy(*args, mode="eval")

        driver(policy, steps=num_timesteps)

        env.close()
        return
    else:
        env = env.envs[0]

    total_reward = 0
    obs = env.reset()

    env.render(mode=render_mode, **kwargs)

    print(env)

    for t in range(num_timesteps):
        action, state = model.predict(obs, deterministic=True)
        if not hasattr(action, "__len__"):
            action = [action]
        obs, reward, done, _info = env.step(action[0])

        if env_normalize:
            reward = env.get_original_reward()

        print(f"t = {t+1}, reward = {reward}")
        total_reward += reward

        # Render the first env of the DummyVecEnv created by setup_environment
        env.render(mode=render_mode, **kwargs)
        if done:
            break

    print(f"Total reward: {total_reward}")

    env.close()
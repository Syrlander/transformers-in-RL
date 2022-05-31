import argparse
import importlib

import rl_thesis.environments.utils as env_utils
import json
from rl_thesis.config_parsing import arg_helpers, overwrite_default_values
from gym.envs import register
from rl_thesis.models.utils import create_model_monitor_dir


def get_argparser():
    parser = argparse.ArgumentParser(prog="rl_thesis train")

    parser.add_argument("model")
    parser.add_argument("environment")
    parser.add_argument("--env_config", default=None)
    parser.add_argument("--eval_env_config", default=None)
    parser.add_argument("--model_config", default=None)
    parser.add_argument("--policy", default="")
    parser.add_argument("--policy_config", default=None)
    parser.add_argument("--rewards_dir", default="train_reward_logs/")
    parser.add_argument("--eval_rewards_dir", default="eval_reward_logs/")

    return parser


def entry_func(args):

    args, help_args = arg_helpers.split_args(args)
    parser = get_argparser()

    parsed, remaining = parser.parse_known_args(help_args or args)

    model_module = importlib.import_module("rl_thesis.models." + parsed.model)
    model_conf = model_module.Config()
    if parsed.model_config:
        overwrite_default_values(parsed.model_config, model_conf)

    is_dreamer = parsed.model == "Dreamer"

    training_rewards_dir = create_model_monitor_dir(parsed.rewards_dir,
                                                    parsed.model,
                                                    parsed.environment,
                                                    parsed.policy)

    eval_rewards_dir = create_model_monitor_dir(parsed.eval_rewards_dir,
                                                    parsed.model,
                                                    parsed.environment,
                                                    parsed.policy)

    env = env_utils.setup_environment(parsed.environment,
                                      n_envs=model_conf.n_envs,
                                      normalize=model_conf.normalize,
                                      monitor_dir=training_rewards_dir,
                                      config_file=parsed.env_config,
                                      is_dreamer_env=is_dreamer)

    eval_env = env_utils.setup_environment(parsed.environment,
                                           normalize=model_conf.normalize,
                                           monitor_dir=eval_rewards_dir,
                                           config_file=parsed.eval_env_config,
                                           is_dreamer_env=is_dreamer)

    pol = None
    policy_kwargs = {}

    # if pol != None:
    try:
        # we just need a reference to the class, so do not need to instantiate the class
        pol_module = importlib.import_module("rl_thesis.policies." +
                                             parsed.policy)
        pol_config = pol_module.Config()
        if parsed.policy_config:
            overwrite_default_values(parsed.policy_config, pol_config)
        policy_kwargs = vars(pol_config)
        pol = pol_module.Policy
    except ModuleNotFoundError:
        # if it is not one of our policies, we assume that it is a policy defined in stable-baselines
        pol = parsed.policy
        if parsed.policy_config:
            with open(parsed.policy_config) as f:
                policy_kwargs = json.load(f)

    model = model_module.Model(model_conf, env, pol, policy_kwargs)
    model.train(eval_env)

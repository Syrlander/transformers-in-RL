import argparse

from rl_thesis.evaluate import evaluate
from rl_thesis.config_parsing import arg_helpers
import importlib


def arg_parser():
    parser = argparse.ArgumentParser(prog="rl_thesis evaluate")

    parser.add_argument("model")
    parser.add_argument("model_filepath")
    parser.add_argument("env_name")
    parser.add_argument("num_timesteps", type=int)
    parser.add_argument("--env_normalize", type=bool, default=False)
    parser.add_argument("--env_config_file", default=None)
    parser.add_argument("--device", default="cpu")

    return parser


def entry_func(args):
    args, help_args = arg_helpers.split_args(args)
    parser = arg_parser()

    parsed, remaining = parser.parse_known_args(help_args or args)

    model_module = importlib.import_module("rl_thesis.models." + parsed.model)
    model_class = model_module.Model

    evaluate(model_class,
             parsed.model_filepath,
             parsed.env_name,
             parsed.num_timesteps,
             env_normalize=parsed.env_normalize,
             env_config=parsed.env_config_file,
             device=parsed.device)

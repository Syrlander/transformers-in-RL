import argparse

from rl_thesis.render import render
from rl_thesis.config_parsing import arg_helpers
import importlib


def arg_parser():
    parser = argparse.ArgumentParser(prog="rl_thesis render")

    parser.add_argument("model")
    parser.add_argument("model_filepath")
    parser.add_argument("env_name")
    parser.add_argument("num_timesteps", type=int)
    parser.add_argument("--render_mode", default="human")
    parser.add_argument("--env_normalize", type=bool, default=False)
    parser.add_argument("--env_config_file", default=None)
    parser.add_argument("--device", type=str, default="cpu")

    return parser


def entry_func(args):
    args, help_args = arg_helpers.split_args(args)
    parser = arg_parser()

    parsed, remaining = parser.parse_known_args(args or help_args)

    # Parse remaining arguments given to the environment render method
    render_types = {
        "frames_freq": int,
        "remove_frame_files": bool
    }

    render_kwargs = arg_helpers.parse_remaining_options(remaining, types=render_types)

    model_module = importlib.import_module("rl_thesis.models." + parsed.model)
    model_class = model_module.Model

    render(
        model_class,
        parsed.model_filepath,
        parsed.env_name,
        parsed.num_timesteps,
        render_mode=parsed.render_mode,
        env_normalize=parsed.env_normalize,
        env_config=parsed.env_config_file,
        device=parsed.device,
        **render_kwargs)

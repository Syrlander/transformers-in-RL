#!/usr/bin/env python3
import argparse
import sys
from .register_envs import register_envs
from rl_thesis import config_parsing


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("script",
                        type=str,
                        help="Name of the script to run.",
                        choices=["train", "evaluate", "plot", "render"])
    return parser


def entry_func():
    # registers all custom envs & policies
    register_envs()

    args, help_args = config_parsing.split_args(sys.argv[1:])
    parser = get_parser()
    # parse args if there are any else try help_args
    parsed, remaining_args = parser.parse_known_args(args or help_args)

    import importlib
    script_module = importlib.import_module("rl_thesis.bin." + parsed.script)

    # Call entry function with remaining arguments
    script_module.entry_func(remaining_args + help_args)

if __name__ == "__main__":
    entry_func()
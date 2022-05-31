import argparse
from rl_thesis.plotting import plot
from rl_thesis.config_parsing import arg_helpers


def arg_parser():
    parser = argparse.ArgumentParser(prog="rl_thesis plot")

    parser.add_argument("plot_name",
                        choices=["window_average_return", "plot_returns"])
    parser.add_argument("monitor_filepath", nargs="+", help="One or more paths to monitor files")
    parser.add_argument("--labels", nargs="+", help="Labels for each monitor plotted, currently only works for window return")
    parser.add_argument("--img_file", default=None)
    parser.add_argument("--baseline", type=float, help="plots a horizontal line at this value")
    parser.add_argument("--baseline_label")
    return parser


def entry_func(args):
    args, help_args = arg_helpers.split_args(args)
    parser = arg_parser()

    parsed, remaining = parser.parse_known_args(args or help_args)

    # Parse any additional arguments as kwargs to the plotting function
    types = {
        "env_ret_min": float,
        "env_ret_max": float,
        "w": int,
        "title": str,
        "time_unit": str,
        "best_models_dir": str,
        "fig_width": float,
        "fig_height": float,
        "img_file": str,
        "ylabel": str
    }
    print(parsed.monitor_filepath)
    plot_kwargs = arg_helpers.parse_remaining_options(remaining, types)
    plot(parsed.plot_name,
         parsed.monitor_filepath,
         img_file=parsed.img_file,
         labels=parsed.labels,
         baseline=parsed.baseline,
         baseline_label=parsed.baseline_label,
         **plot_kwargs)

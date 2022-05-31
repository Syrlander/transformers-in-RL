import pandas as pd
import sys
import argparse
from stable_baselines3.common.results_plotter import window_func
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("input_file", help="csv file to aggregate")
    parser.add_argument("output_file", help="path to save csv file")
    parser.add_argument("window", help="Size of the window to aggregate over", type=int)
    parser.add_argument("timesteps_between_evals", help="The number of time steps that was between each evaluation, this will be used to fill the timesteps column", type=int)
    parser.add_argument("columns", nargs="*", help="Columns to aggregate")
    parser.add_argument("--eval_at_zero", help="Denotes whether the first evaluation was done at the 0th timestep", default=False, action="store_true")
    parser.add_argument("--skip_rows", help="Number of rows in input files to skip", default=0, type=int)
    return parser

def window_mean(var, window_size):
    indices = []
    start = 0
    while start < len(var):
        indices.append(list(range(start, start+window_size)))
        start += window_size
    windows = var[np.array(indices)]
    return windows.mean(axis=1)

def main():
    parser = get_parser()

    parsed = parser.parse_args(sys.argv[1:])

    input_df = pd.read_csv(parsed.input_file, skiprows=parsed.skip_rows)
    timesteps_between_evals = parsed.timesteps_between_evals
    first_time_step = 0 if parsed.eval_at_zero else parsed.timesteps_between_evals
    total_evals = int(len(input_df)/parsed.window)
    final_time_step = total_evals * timesteps_between_evals + (1 if not parsed.eval_at_zero else 0)
    timesteps = range(first_time_step, final_time_step, timesteps_between_evals)
    
    output_df = pd.DataFrame(columns=["train_timestep"] + parsed.columns)
    output_df["train_timestep"] = timesteps
    for col in parsed.columns:
        agg_input = np.array(input_df[col])
        y_win = window_mean(agg_input, parsed.window)
        output_df[col] = y_win
    
    output_df.to_csv(parsed.output_file, index=False)



if __name__ == "__main__":
    main()
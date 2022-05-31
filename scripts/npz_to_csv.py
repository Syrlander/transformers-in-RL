import numpy as np
import pandas as pd
import argparse
import sys

def get_parser():
    parser = argparse.ArgumentParser(prog="npz_to_csv")

    parser.add_argument("input_file", help="npz file to convert to csv")
    parser.add_argument("output_file", help="path to save csv file")

    return parser


def main():
    parser = get_parser()
    parsed = parser.parse_args(sys.argv[1:])

    input_file = parsed.input_file
    output_file = parsed.output_file

    arr = np.load(input_file)
    keys = arr.keys()
    df = pd.DataFrame(columns=list(keys))
    for key in keys:
        val = arr[key]
        if len(val.shape) > 1:
            val = val.mean(axis=1)
        df[key] = val
    
    df.rename({"results":"mean_return", "timesteps":"train_timestep"}, inplace=True, axis="columns")
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
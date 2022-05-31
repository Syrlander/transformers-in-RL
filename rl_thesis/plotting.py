from argparse import ArgumentError
from stable_baselines3.common.results_plotter import window_func
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def window_average_return(episodes,
                          returns,
                          env_ret_min=-1e10,
                          env_ret_max=1e10,
                          w=100,
                          title=None,
                          time_unit="episodes",
                          best_models_dir=None,
                          label=None):
    """
    Plot a rolling window average of episode returns, along with 1 std. boundaries.

    Args:
        episodes: episode numbers
        returns: episode returns
    Kwargs:
        env_ret_min (default=-1e10): environments min. return
        env_ret_max (default=1e10): environments max. return
        w (default=100): window size
        title (default=None): plot title, not set if None
        best_models_dir: if specified (and dir. exists) assume best model file name format as: MODEL_NAME_TIMESTEP.EXT
                         where "MODEL_NAME" and "EXT" can be anything. The TIMESTEP is then used to draw a vertical
                         line indicating when the current best model was saved.
    """
    x_win, y_win_avg = window_func(episodes, returns, w, np.mean)
    _, y_win_std = window_func(episodes, returns, w, np.std)

    plt.plot(x_win, y_win_avg, zorder=100, label=label)
    plt.fill_between(x_win,
                     np.clip(y_win_avg - y_win_std, env_ret_min, env_ret_max),
                     y2=np.clip(y_win_avg + y_win_std, env_ret_min,
                                env_ret_max),
                     alpha=0.4,
                     zorder=90)

    if best_models_dir:
        best_models_dir = Path(best_models_dir)
        best_xs = []
        for best_model_file in best_models_dir.glob("*"):
            timestep = int(best_model_file.stem.split("_")[-1])
            best_xs.append(timestep)

        plt.vlines(best_xs,
                   np.max([env_ret_min,
                           np.min(y_win_avg - y_win_std - 10)]),
                   np.min([env_ret_max,
                           np.max(y_win_avg + y_win_std + 10)]),
                   linestyle="--",
                   colors="m",
                   label="best models",
                   zorder=50)
        plt.legend()

    plt.title(title)
    if time_unit == "episodes":
        plt.xlabel("Episode #")
    elif time_unit == "timestep":
        plt.xlabel("Timestep #")
    elif time_unit == "train_timestep":
        plt.xlabel("Training time step #")
    plt.ylabel(f"Rolling Average Return ($w = {w}$)")
    plt.grid(zorder=1)


def plot_returns(episodes, returns, title=None):
    """
    Plot episode returns as a scatter/line plot

    Args:
        episodes: episode numbers
        returns: episode returns
    
    Kwargs:
        title (default=None): plot title, not set if None
        img_file (default=None): filepath location of image to export. Don't save if None
    """
    plt.plot(episodes, returns, linewidth=0.4, zorder=95)
    plt.scatter(episodes, returns, c="r", s=3, zorder=95)

    plt.title(title)
    plt.xlabel("Episode #")
    plt.ylabel(f"Return")
    plt.grid(zorder=1)


def mean_pad(arrs):
    """
    Pad a Python list of numpy arrays of non-equal length with the mean
    of the existing values that extends further than the shorter arrays
    """
    max_arr_len = np.max([arr.shape[0] for arr in arrs])
    new_arrs = np.array([
        np.pad(arr, (0, max_arr_len - arr.shape[0]), constant_values=np.nan)
        for arr in arrs
    ])
    nan_means = np.tile(np.nanmean(new_arrs, axis=0), (new_arrs.shape[0], 1))
    nan_mask = np.isnan(new_arrs)
    new_arrs[nan_mask] = nan_means[nan_mask]
    return new_arrs

def get_reward_col_name(df):
    possible_col_names = ["r", "mean_return", "results"] 
    col_name = [name for name in possible_col_names if name in df.columns]
    if not len(col_name) == 1:
        raise Exception(f"Found {len(col_name)} possible columns to use as mean return from {list(df.columns)}")
    return col_name[0]

def load_returns_log(monitor_filepath, time_unit):
    monitor_filepath = Path(monitor_filepath)

    if monitor_filepath.is_dir():
        if not time_unit == "episodes":
            raise NotImplementedError(
                "Case of using num. timesteps for x-axis is not yet supported for multiple environments, as each environment episode may end at different timesteps."
            )

        print(f"Loading returns from dir.: {monitor_filepath}")
        # If the monitor path is to a directory, we take an average of the returns
        # for each episode. This is useful for e.g. PPO which can work on multiple
        # environments in parallel
        data_frames = [
            pd.read_csv(file, skiprows=1)
            for file in monitor_filepath.glob("*.monitor.csv")
        ]

        longest_episode_idx = np.argmax([df.shape[0] for df in data_frames])
        episodes = data_frames[longest_episode_idx].index.to_numpy()

        # Not necessarily the same length, i.e. different num. episodes
        # Pad missing episodes with mean over the existin episodes to leave
        # the actual mean unchanged
        col_name = get_reward_col_name(df)
        r_arrs = np.mean(mean_pad([df[col_name].to_numpy() for df in data_frames]),
                         axis=0)
        return episodes, r_arrs
    else:
        print(f"Loading returns from file: {monitor_filepath}")
        try:
            df = pd.read_csv(monitor_filepath, skiprows=1)
            col_name = get_reward_col_name(df)
        except:
            df = pd.read_csv(monitor_filepath)
            col_name = get_reward_col_name(df)

        if time_unit == "episode":
            # Use episode nums.
            x = df.index.to_numpy() + 1
        elif time_unit == "timestep":
            # Use num. timesteps
            x = np.cumsum(df["l"].to_numpy())
        elif time_unit == "train_timestep":
            # use number of time steps trained for, good for evaluation stuff
            x = df["train_timestep"].to_numpy()

        return (x, df[col_name].to_numpy())


def check_time_unit(time_unit):
    if not time_unit in ["train_timestep","episode","timestep"]:
        raise ArgumentError("time_unit", f"time unit must be one of train_timestep, episode, timestep, but got {time_unit}")

def plot(plot_name,
         monitor_filepath,
         time_unit="episode",
         img_file=None,
         fig_width=10,
         fig_height=6,
         labels=None,
         baseline=None,
         baseline_label=None,
         ylabel=None,
         **kwargs):
    """
    Entrypoint of calling 'rl_thesis plot'

    Kwargs:
        time_unit: what unit to use along x-axis. Option:
            train_timestep: will use column train_timesteps to show for how long the model has trained (good for plotting evaluation scores)
            episode: plots number of episodes
            timestep: plots cumulative number of timesteps run
    """
    check_time_unit(time_unit)   
    plt.figure(figsize=(fig_width, fig_height))
    if labels is None:
        labels = [None] * len(monitor_filepath)
    assert len(labels) == len(monitor_filepath), "Number of labels must equal number of monitors to plot!"
    x_max = 0 # used to see how far to draw baseline line
    for filepath, label in zip(monitor_filepath, labels):
        xs, returns = load_returns_log(filepath, time_unit)
        if max(xs) > x_max:
            x_max = max(xs)


        if plot_name == "window_average_return":
            window_average_return(xs, returns, time_unit=time_unit, label=label, **kwargs)
        elif plot_name == "plot_returns":
            plot_returns(xs, returns, **kwargs)
        else:
            raise ArgumentError(f"Got invalid plot_name argument: '{plot_name}'")
    if ylabel:
        plt.ylabel(ylabel)
    if baseline:
        plt.hlines([baseline],
                   0,
                   x_max, 
                   linestyle="--",
                   colors="m",
                   label=baseline_label,
                   zorder=50)
    if labels[0]:
        plt.legend()
    if img_file:
        plt.tight_layout()
        plt.savefig(img_file)
    else:
        plt.show()

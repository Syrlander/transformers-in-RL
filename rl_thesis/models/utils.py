from datetime import datetime
from pathlib import Path


def create_model_monitor_dir(rewards_dir, model_name, env_name, policy_name):
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    path = Path(rewards_dir, env_name, f"{model_name}-{policy_name}_{now_str}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_model_save_dir(models_dir, model_name, env_name, policy):
    # cast policy to str, as we might use our own policy implementation
    now_str = datetime.now().strftime("%d-%m-%Y_%H-%M")
    path = Path(models_dir, env_name, f"{model_name}-{str(policy)}-{now_str}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_metrics_dict_to_csv(metrics, csv_filepath, is_first_row=False):    
    if is_first_row:
        # Create file if doesn't exist - i.e. on first write
        fp = csv_filepath.open("w")
    else:
        fp = csv_filepath.open("a")

    # write header row
    if is_first_row:
        headers = ",".join(metrics.keys())
        fp.write(headers)
        fp.write("\n")

    # write data row
    fp.write(",".join(map(str, metrics.values())))
    fp.write("\n")

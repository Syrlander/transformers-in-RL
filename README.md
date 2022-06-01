# Master Thesis: Applications of Transformer Networks in Reinforcement Learning
This repository serves as storage for all code relating to our Master thesis project.

It should be noted that configuration files may *not* reflect the final hyperparameters of experiments described in the report and we refer to the Appendix of our report for a detailed overview of all hyperparameters.

## Code Structure
The following provides a general overview of the code:
* `configs`: JSON configuration files of all models and environments. Structured such that each environment has it's own directory, with all model configs relating to the environment inside it. An environment directory may also contain an `env_config` directory, containing the environment configuration files (primarily for Numpad environment).
* `rl_thesis`: directory of all files included in the package.
    * `bin`: entry points for all CLI arguments. E.g. `train.py` maps to the `train` CLI argument, etc.
    * `config_parsing`: parsing utilities for reading in CLI options and/or additional model/algorithm configurations from JSON files.
    * `environments`: implementations of all custom environments and model specific wrappers.
        * Includes: All three Numpad versions, Mountain Car without velocity, and Mountain Car with image observations.
    * `dreamer`: implementation of our port of the DreamerV2 model (https://github.com/danijar/dreamerv2).
    * `gated_transformer`: implementation of our single-CPU/single-GPU version of V-MPO with options for using Vanilla Transformers, TrXL, TrXL-I and GTrXL for the shared core.
        * Implementation is based upon modifications of code from: https://github.com/jsikyoon/V-MPO_torch.
    * `models`: directory of all models available via the CLI, each following our API design such that they can be trained, loaded, rendered, and evaluated.
    * `policies`: policy files relating to our DQN/DRQN implementation, along with a utility implementation of a fully-connected neural network used in other models.
    * `tests`: test cases relating to variuous parts of the codebase, implemented using `pytest`
    * `algorithms`: random agent/policy compatible with the stable-baselines3 interface and the replay experience buffer used throughout multiple models.
    * `utils`: general utility functions for capturing various PyTorch metrics and individual layers.
    * `evaluate.py`: functions used to evalute different models.
    * `plotting.py`: functions used to plot graphs of different model outputs and metrics.
    * `render.py`: functions used to render model behavior (e.g. if loading a trained mdoel) in a given environment.
    * `config.py`: base configuration class used by all subsequent model configuration classes within `models/`.
* `slurm_scripts`: all scripts relating to starting Slurm jobs on the cluster and mounting the ERDA storage server (https://erda.dk/), used for storing output and model files. Each model has it's own directory containing a script to start a job for running the model in a specific environment.
* `scripts`: minor scripts used for data aggregations/analysis.
    * `npz_to_csv.py`: convert numpy compressed files to csv.
    * `aggregate_eval_results.py`: 
* `requirements.txt`: file of all python3.6 dependencies required for running the project.

### API design
Our framework has been designed so new models can easiler be added, such that they are compatible with the CLI. The design is best explained by going over the steps required to add a new model.

0. All models are defined within `rl_thesis/models`. Where sub-directory names match those of the model names, denoted as `MODEL_NAME`, exposed through the CLI.
1. Create a nwe sub-directory under `rl_thesis/models` with name `MODEL_NAME`. The directory should contain:
    * `config.py`: should include a `Config` class inheriting from `BaseModelConfig`, where any attributes constitue hyperparameters of the model, along with their default values.
    * `model.py`: should contain at least one class inheriting from `BaseModel`, requiring two methods to be implemented:
        * `load(path, env=None, device="cpu")`: method for loading the model from a given file.
        * `train(eval_env)`: method for training the model, where `eval_env` is the environment used for evaluation runs inbetween training.
    * `__init__.py`: imports the model class and configuration classes and exposes them as `Model` and `Config` respectively.
    * Each model can have separate files located within the `rl_thesis` directory, such as the ones for Dreamer, V-MPO/GTrXL, and DRQN. Such that the files under `rl_thesis/models` focus on implementing the training loop and loading, whilst the separate files provide the model implementation itself.
2. Once setup the model is available in the CLI as `MODEL_NAME`, e.g. to train a model a command could be:
```
(venv)> rl_thesis train MODEL_NAME ENV_NAME --model_config JSON_CONFIG_FILE
```
which would train the `MODEL_NAME` model on the `ENV_NAME` environment using hyperparameters from the `JSON_CONFIG_FILE`. Note that hyperparameters from a JSON file only overwrite defaults of the model config, if a hyperparameter is not included in the JSON file the default value is used instead.

With this API design models are independent of each other and the environments, where a large number of different experiments can be constructed directly from the CLI without changing the codebase.

For a full description of all CLI arguments and options be refer to the below sections.

### Installation
The project has been developed using Python 3.6.8.

During development we have primarily relied upon the usage of Python virtual environments, as such a development environment can be created by:
1. Install Python 3.6.8 using your systems package manager.
2. Download repository:
```
> git clone https://github.com/Syrlander/transformers-in-RL.git
> cd transformers-in-RL
```
3. Create and activate new virtual environment
```
transformers-in-RL> python3.6 -m venv venv
transformers-in-RL> source ./venv/bin/activate
```
4. Install dependencies
```
(venv)> pip install -r requirements.txt
```
If you encounter issues during installation of dependencies, it is recommended to install and resolve issues of each package one-by-one.

5. (Optional) Install Atari ROMs, if running the original Dreamer model, by following the instructions of: https://github.com/openai/atari-py#roms.

6. Install our package exposing the CLI
```
(venv)> pip install .
```

### Command-Line Interface (CLI)
Here we provide a full description of each CLI argument and their options.

Note that `MODEL_NAME` refers to any model within `rl_thesis/models/` and `ENV_NAME` is the name of any environment available through OpenAI gym or any variation of the below Numpad environments.

#### Training
```
(venv)> rl_thesis train MODEL_NAME ENV_NAME
```

Options:
* `--env_config`: JSON configuration file pass to initialization of the environment. Can be used to specify the Numpad configurations with sequence length, number of tiles, etc.
* `--eval_env_config`: JSON configuration file pass to initialization of the evaluation environment.
* `--model_config`: JSON configuration file of model hyperparameters.
* `--policy`: Name of policy (policy network) to use along with the model. E.g. in the case of DQN/DRQN this can be specified to switch between MLP and CNN based policies.
* `--policy_config`: JSON configuration file of policy (policy network) hyperparameters, e.g. number of network layers, layer sizes, etc.
* `--rewards_dir` (default: `train_reward_logs/`): Path to directory of where to write episode returns of the training environment.
* `--eval_rewards_dir` (default: `eval_reward_logs/`): Path to directory of where to write episode returns of the evaluation environment.

#### Evaluation
```
(venv)> rl_thesis evaluate MODEL_NAME SAVE_MODEL_FILEPATH ENV_NAME NUM_TIMESTEPS
```
Where `SAVE_MODEL_FILEPATH` is the filepath to a model saved during training (e.g. either via the model obtaining a new highest return in the evaluation environment or by the checkpointed models), and `NUM_TIMESTEPS` being the number of tiem steps to evaluate the model for.

Options:
* `--env_normalize`: whether or not to normalize observations/rewards of the environment.
* `--env_config_file`: JSON configuration file pass to initialization of the environment. Can be used to specify the Numpad configurations with sequence length, number of tiles, etc.
* `--device`: device of where to evaluate the model, either `cpu` or `cuda`

#### Rendering
```
(venv)> rl_thesis render MODEL_NAME SAVE_MODEL_FILEPATH ENV_NAME NUM_TIMESTEPS
```
Where `SAVE_MODEL_FILEPATH` is the filepath to a model saved during training (e.g. either via the model obtaining a new highest return in the evaluation environment or by the checkpointed models), and `NUM_TIMESTEPS` being the number of tiem steps to evaluate the model for.

Options:
* `--render_mode`: render mode of the environment. Note that this is environment dependent.
* `--env_normalize`: whether or not to normalize observations/rewards of the environment.
* `--env_config_file`: JSON configuration file pass to initialization of the environment. Can be used to specify the Numpad configurations with sequence length, number of tiles, etc.
* `--device`: device of where to evaluate the model, either `cpu` or `cuda`

#### Plotting
```
(venv)> rl_thesis plot PLOT_NAME MONITOR_FILEPATH
```
Where `PLOT_NAME` specifies the plotting function to use, either (`window_average_return` or `plot_returns`). `MONITOR_FILEPATH` is the filepath to a monitor file containing episode returns created during training/evaluation under the `rewards_dir` or `eval_rewards_dir` (see Training subsection above).

Options:
* `--labels`: labels to use for the plot, passed as arguments to the plotting function.
* `--img_file`: filepath of where to store the resulting plot.
* `--baseline`: value of horizontal line to be plotted for baseline models.
* `--baseline_label`: name of the baseline model to use in legends.

Note that any additional options are pass to the chosen plotting function.

## Numpad Environments
Included in the codebase is an implementation of the Numpad environment (the long-term memory task), with three separate versions:
1. `numpad_discrete-v1`: Discrete control version, with dicrete actions.
2. `numpad_continuous-v1`: Continuous control version, with continuous actions.
3. `numpad_`: Mixed version, with discrete actions mapping onto specific continuous actions.

Code relating to the Numpad environment implementation can be found under: `rl_thesis/environments/`. For a stand-alone implementation of these Numpad environments as a python package we refer to our other repository: https://github.com/Syrlander/numpad-gym.

### Testing
As we provide our own implementations of the Numpad discrete and continuous environments, we have created an associated suite of unit and integration tests which can be ran using.

Run all tests relating to Numpad environments:
```
(venv)> pytest rl_thesis/tests/environment_tests/test_numpad.py
```

Run test coverage:
```
(venv)> coverage run -m pytest -v rl_thesis/tests/environment_tests/test_numpad.py && coverage report -i
```

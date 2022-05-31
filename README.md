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
    * `algorithms`: individual custom policies (only includes random model) and the replay experience buffer used throughout multiple models. (MOVE THIS UNDER `policies` IF WE HAVE TIME - SEEMS A BIT MESSY?!)
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


### Installation
The project has been developed using Python 3.6.8.

During development we have primarily relied upon the usage of Python virtual environments, as such a development environment can be created by:
1. Install Python 3.6.8 using your systems package manager.
2. Download repository:
```
> git clone INSERT_REPO_URL
> cd INSERT_REPO_NAME
```
3. Create and activate new virtual environment
```
INSERT_REPO_NAME> python3.6 -m venv venv
INSERT_REPO_NAME> source ./venv/bin/activate
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


## Numpad Environments
Included in the codebase is an implementation of the Numpad environment (the long-term memory task), with three separate versions:
1. Discrete control version, with dicrete actions.
2. Continuous control version, with continuous actions.
3. Mixed version, with discrete actions mapping onto specific continuous actions.

Code relating to the Numpad environment implementation can be found under: `rl_thesis/environments/`

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




Github Cleanup ToDo:
* [ ] README:
  * [ ] Get rid of the `algorithms` directory under `rl_thesis`
  * [ ] How things can be ran from CLI and the different arguments (describe what they mean)
  * [x] General project structure of where models, environments, and configuration files are located (and which ones we have)
    * [x] Note that configuration files may not reflect the final hyperparameters of experiments and we refer to the Appendix of our report for the actual parameters used.
  * [x] Tests and how to run them. (pytest on the tests directory or on single files or classes/methods within single files/classes.)
  * [x] requirements.txt, how to install virtual environment and python version (due to the slurm cluster)
* [ ] Remove files/directories relating to other Dreamer implementations.
* [x] Create new repository
  * [ ] Insert the new repo. name and url in the above instructions where needed (search for "INSERT_REPO_URL" and "INSERT_REPO_NAME")
* [x] Describe the different branches
* [ ] Remove `example_out_for_sharing.py` under `rl_thesis/dreamer/`

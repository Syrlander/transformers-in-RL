from rl_thesis.models import BaseModel
from rl_thesis.algorithms.random import Random
from rl_thesis.models.utils import create_model_save_dir
from pathlib import Path


class RandomModel(BaseModel):
    def __init__(self, conf, env, _policy, _policy_kwargs):
        self.conf = conf
        self.env = env
        self.model = Random("RandomPolicy", env)

    @classmethod
    def load(cls, path, env=None, device="cpu"):
        return Random("RandomPolicy", env)

    def train(self, eval_env):
        raise NotImplementedError("Random model does not support training, as it just performs random actions.")

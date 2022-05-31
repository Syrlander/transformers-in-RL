from rl_thesis.models import BaseModel
from rl_thesis.models.utils import create_model_save_dir
from .config import Config
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from pathlib import Path


class PPOModel(BaseModel):
    def __init__(self, conf : Config, env, policy, policy_kwargs):
        super().__init__(conf, env, policy, policy_kwargs)

        self.conf = conf
        self.policy = policy
        self.env = env

        self.model = PPO(
            policy,
            env,
            learning_rate=conf.learning_rate,
            n_steps=conf.n_steps,
            batch_size=conf.batch_size,
            n_epochs=conf.n_epochs,
            gamma=conf.gamma,
            gae_lambda=conf.gae_lambda,
            ent_coef=conf.ent_coef,
            device=conf.device,
            policy_kwargs=policy_kwargs,
            verbose=conf.verbose)

    @classmethod
    def load(cls, path, env=None, device="cpu"):
        return PPO.load(path, env=env)

    def train(self, eval_env):
        model_save_path = create_model_save_dir(
            self.conf.models_dir,
            "PPO",
            self.env.envs[0].spec.id,
            self.policy
        )

        self.model.learn(
            total_timesteps=self.conf.total_timesteps,
            log_interval=4,
            eval_env=eval_env,
            eval_freq=self.conf.eval_freq,
            n_eval_episodes=self.conf.n_eval_episodes,
            eval_log_path=str(model_save_path))

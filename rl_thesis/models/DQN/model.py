from rl_thesis.models import BaseModel
from rl_thesis.models.utils import create_model_save_dir
from .config import Config
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from rl_thesis.environments.utils import setup_environment
import json
from rl_thesis.models import utils
from rl_thesis.config import BaseConfig
class DQNModel(BaseModel):
    def __init__(self, conf : Config, env, policy, policy_kwargs):
        super().__init__(conf, env, policy, policy_kwargs)

        self.conf = conf
        self.policy = policy
        self.env = env
        self.log_dir = utils.create_model_monitor_dir(self.conf.log_dir, "DQN", self.env.envs[0].spec.id, self.policy)

        model_config_json_obj = self.conf.to_json_serializable_dict()
        json.dump(model_config_json_obj, (self.log_dir / "model_config.json").open("w"), indent=4)

        policy_config = BaseConfig()
        [setattr(policy_config, k, v) for k, v in policy_kwargs.items()]
        policy_config_json_obj = policy_config.to_json_serializable_dict()
        json.dump(policy_config_json_obj, (self.log_dir / "policy_config.json").open("w"), indent=4)

        self.model = DQN(
            policy,
            env,
            batch_size=conf.batch_size,
            buffer_size=conf.buffer_size,
            learning_starts=conf.learning_starts,
            gamma=conf.gamma,
            target_update_interval=conf.target_update_interval,
            train_freq=conf.train_freq,
            gradient_steps=conf.gradient_steps,
            exploration_fraction=conf.exploration_fraction,
            exploration_final_eps=conf.exploration_final_eps,
            learning_rate=conf.learning_rate,
            device=conf.device,
            policy_kwargs=policy_kwargs,
            verbose=conf.verbose)

    @classmethod
    def load(cls, path, env=None, device="cpu"):
        return DQN.load(path, env=env)

    def train(self, eval_env):
        model_save_path = create_model_save_dir(
            self.conf.models_dir,
            "DQN",
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

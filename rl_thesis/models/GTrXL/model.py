from struct import unpack
from rl_thesis.models import BaseModel
from rl_thesis.models.GTrXL.config import Config
from rl_thesis.gated_transformer import VMPO
from rl_thesis.algorithms.utils.replay_memory import ReplayMemory, Transition
from tqdm import tqdm
import numpy as np
import torch
from rl_thesis.models import utils


class VMPOMemory(ReplayMemory):

    BATCH_KEYS = {
        "state",
        "action",
        "reward",
        "step_num",
        "done"
    }

    def push(self, state, action, next_state, reward, step_num, done):
        """
        Remarks:
            Numpy arrays and/or Python lists are cast to torch Tensors within the models
            so always set is_torch to False in VMPO memory
        """
        # Track previous action and rewards for use in ActorCritic.act
        self.prev_action = action
        self.prev_reward = reward

        return super().push(state, action, next_state, reward, step_num, done, is_torch=False)

    def sample_sequential_steps(self, batch_size, n_time_steps, replace=False):
        transitions = super().sample_sequential_steps(batch_size, n_time_steps, replace)

        if transitions == []:
            raise ValueError(f"Not enough memory samples to construct batch! Memory contains {self.current_episode + 1} with a total of {self.n_transitions} time steps")

        # Transform transitions from List[List[Transition]] to Dict[str, List[List[int]]]
        batch = {key: [] for key in self.BATCH_KEYS}
        
        for trajectory in transitions:
            unpacked_trajectory = {key: [] for key in self.BATCH_KEYS}
            
            for transition in trajectory:
                for key in self.BATCH_KEYS:
                    unpacked_trajectory[key].append(getattr(transition, key))

            for key in self.BATCH_KEYS:
                batch[key].append(unpacked_trajectory[key])

        tmp_value = batch["step_num"]
        batch["timestep"] = tmp_value
        return batch

    def clear(self):
        self.__init__(self.capacity)


class GTrXLModel(BaseModel):
    def __init__(self, conf : Config, env, policy, policy_config):
        super().__init__(conf, env, policy, policy_config)
        self.conf = conf
        self.env = env

        action_dim = self.env.action_space.n
        observation_dim = self.env.observation_space.shape[0]

        self.memory = VMPOMemory(self.conf.capacity)
        self.eval_memory = VMPOMemory(self.conf.capacity)
        self.agent = VMPO(
            self.conf.state_rep,
            action_dim,
            observation_dim,
            n_head=self.conf.n_head,
            n_layer=self.conf.n_layer,
            encoder=self.conf.encoder,
            mlp_encoder_layers=self.conf.mlp_encoder_layers,
            mlp_encoder_dim=self.conf.mlp_encoder_dim,
            n_latent_var=self.conf.n_latent_var,
            mem_len=self.conf.mem_len,
            init_alpha=self.conf.init_alpha,
            init_eta=self.conf.init_eta,
            eps_eta=self.conf.eps_eta,
            lr=self.conf.lr,
            betas=self.conf.betas,
            gamma=self.conf.gamma,
            eps_alpha=self.conf.eps_alpha,
            K_epochs=self.conf.K_epochs,
            device=self.conf.device
        ).to(self.conf.device)

        print(self.agent)

        self.model_name = f"GTrXL-{self.conf.state_rep}"
        self.model_save_dir = utils.create_model_save_dir(
            self.conf.models_dir,
            self.model_name, 
            self.env.envs[0].spec.id,
            "VMPO")
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        metrics_log_dir = utils.create_model_monitor_dir(
            self.conf.log_dir, 
            self.model_name,
            self.env.envs[0].spec.id, 
            "VMPO")
        metrics_log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log_file = metrics_log_dir / "train_metrics.csv"

        self.is_first_log = True

    @classmethod
    def load(self, path, env=None, device="cpu"):
        agent = torch.load(path, map_location=device)
        # device is used in agent to move tensors, this overwrites the saved device to the new device
        agent.device = device
        return agent

    def train(self, eval_env):
        """
        Episode returns are saved implicitly via environment wrappers
        """
        best_model_return = -np.inf

        total_timesteps, eval_timestep = 0, 0
        update_episode = 0

        for _ in tqdm(range(self.conf.max_episodes), desc="Training"):
            obs = self.env.reset()[0]

            for t in range(self.conf.max_episode_timesteps):
                action = self.agent.policy_old.act(t, obs, self.memory)

                # Note we are using wrapped stable-baselines vectorized environments
                # So we wrap actions in lists and unpack, as we only consider as single env.
                obs, reward, done, _info = self.env.step([action])
                obs, reward, done = obs[0], reward[0], done[0]

                eval_timestep += 1
                total_timesteps += 1

                self.memory.push(obs, action, None, reward, t, done)

                if done:
                    update_episode += 1
                    break

            # Model learning
            if update_episode >= self.conf.update_episode:
                # print(f"batch size: {self.conf.batch_size}, n = {self.conf.unroll_length}")
                # print(f"length of memory: {len(self.memory.memory)}")
                batch = self.memory.sample_sequential_steps(self.conf.batch_size, self.conf.unroll_length)
                # print(f"update batch: {batch}")
                metrics = self.agent.update(batch)

                metrics["timestep"] = total_timesteps
                utils.add_metrics_dict_to_csv(
                    metrics,
                    self.metrics_log_file,
                    is_first_row=self.is_first_log)

                # Ensure header is only written once
                if self.is_first_log:
                    self.is_first_log = False

                # Clear training memory
                self.memory.clear()
                update_episode = 0

            # Training evaluation
            if eval_timestep >= self.conf.eval_freq:
                eval_timestep = 0
                eval_returns = self.eval(eval_env)

                # Save best model
                eval_avg_return = np.mean(eval_returns)
                if eval_avg_return > best_model_return:
                    if self.conf.verbose:
                        print(f"found new best model with avg. return: {eval_avg_return} (old best return: {best_model_return})")
                    model_file = self.model_save_dir / f"{self.model_name}_{total_timesteps}.model"
                    torch.save(self.agent, model_file)
                    best_model_return = eval_avg_return

        self.env.close()

    def eval(self, eval_env):
        """
        Episode returns are saved implicitly via environment wrappers
        """
        self.agent.policy_old.eval()

        eval_eps_return, eval_returns = 0, [ ]

        for _ in tqdm(range(self.conf.n_eval_episodes), desc="Evaluation"):
            obs = eval_env.reset()[0]

            for t in range(self.conf.max_episode_timesteps):
                action = self.agent.policy_old.act(t, obs, self.eval_memory)

                obs, reward, done, _info = eval_env.step([action])
                obs, reward, done = obs[0], reward[0], done[0]

                eval_eps_return += reward

                self.eval_memory.push(obs, action, None, reward, t, done)

                if done:
                    eval_returns.append(eval_eps_return)
                    eval_eps_return = 0
                    break

        self.eval_memory.clear()
        self.agent.policy_old.train()

        return np.array(eval_returns)


if __name__ == "__main__":
    print("GTrXL debugging start")

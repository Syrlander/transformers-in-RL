from distutils.log import info
from opcode import stack_effect
from rl_thesis.models import BaseModel

from rl_thesis.policies.torch_policy import TorchPolicy
from .config import Config
from rl_thesis.algorithms.utils import ReplayMemory, Transition
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from tqdm import tqdm
import math
import random
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import os
import importlib
from pathlib import Path
from copy import deepcopy
from pytorch_model_summary import summary
from rl_thesis.models.utils import create_model_save_dir, create_model_monitor_dir
import json
from rl_thesis.models import BaseModelConfig as BaseConfig


class TorchDQNModel(BaseModel):
    def __init__(self, conf: Config, env: VecEnv, policy, policy_config={}):
        super().__init__(conf, env, policy, policy_config)
        self.conf = conf
        self.policy = policy
        self.env = env
        self.memory = ReplayMemory(
            self.conf.replay_total_timesteps,
            prioritized_replay_factor=self.conf.prioritized_replay_factor,
            priority_min_reward=self.conf.priority_min_reward,
        )
        self.policy_net: TorchPolicy = policy(**policy_config).to(conf.device)
        if self.conf.verbose:
            summary_input = torch.rand(size=(1,1) + self.env.observation_space.shape).to(self.conf.device)
            summary(self.policy_net, summary_input, print_summary=True)
        self.target_net: TorchPolicy = policy(**policy_config).to(conf.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_net.toggle_optimization_step(True)
        self.training = True
        self.n_updates = 0
        self.n_steps = 0
        self.best_eval_reward = -np.inf
        self.save_path = create_model_save_dir(
            self.conf.models_dir, 
            "torch_DQN", 
            self.env.envs[0].spec.id, 
            self.policy.__module__.split('.')[-1],
            )
        self.log_dir = create_model_monitor_dir(
            self.conf.log_dir,
            "torch_DQN",
            self.env.envs[0].spec.id,
            self.policy.__module__.split('.')[-1],
        )
        self.save_path.mkdir(parents=True, exist_ok=True)
        optimizer_class = getattr(optim, self.conf.optimizer)
        self.optimizer = optimizer_class(self.policy_net.parameters(),
                                         lr=self.conf.learning_rate)

        model_config_json_obj = self.conf.to_json_serializable_dict()
        json.dump(model_config_json_obj, (self.log_dir / "model_config.json").open("w"), indent=4)

        policy_conf = BaseConfig()
        [setattr(policy_conf, k, v) for k, v in policy_config.items()]
        policy_config_json_obj = policy_conf.to_json_serializable_dict()
        json.dump(policy_config_json_obj, (self.log_dir / "policy_config.json").open("w"), indent=4)

        if self.conf.decay_function == self.conf.DECAY_EXP:
            self.decay_function = self.get_decay_exp(
                self.conf.eps_start,
                self.conf.eps_end,
                self.conf.eps_decay,
            )
        elif self.conf.decay_function == self.conf.DECAY_LINEAR:
            self.decay_function = self.get_decay_linear(
                self.conf.eps_start,
                self.conf.eps_end,
                self.conf.eps_end_step,
            )
        else:
            raise ValueError(
                f"Configured decay function {self.conf.decay_function} is invalid"
            )

    @classmethod
    def load(self, path, env=None, device="cpu"):
        policy_net = torch.load(path, map_location=device)
        policy_net.device = device
        dummy_input = torch.rand((1, 1) + env.envs[0].observation_space.shape).to(device)
        summary(policy_net, dummy_input, print_summary=True)
        return policy_net

    def train(self, eval_env):
        episode_count = 0
        done = False
        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float)
        episode_len = 0
        episode_reward = 0
        losses = []
        episode_rewards = []
        episode_lens = []
        for _ in tqdm(range(int(self.conf.total_timesteps)),
                      disable=not self.conf.verbose or True):
            # Select and perform an action
            action = self._select_action(state, self.n_steps)
            next_state, reward, done, info = self.env.step(
                np.array([action.item()]))
            reward = torch.tensor(reward)

            next_state = torch.tensor(next_state, dtype=torch.float)
            episode_reward += reward[0]
            # If episode is done, set state to None before saving
            if done:
                terminal_state = torch.tensor(
                    [info[0]["terminal_observation"]], dtype=torch.float)

                self.memory.push(state, action, terminal_state, reward,
                                 episode_len, False)

                aux_state = (torch.zeros(next_state.shape) +
                             self.conf.final_state_value)
                self.memory.push(terminal_state, action, aux_state,
                                 torch.Tensor([0.]), episode_len, True)
            else:
                self.memory.push(state, action, next_state, reward,
                                 episode_len, False)
            state = next_state

            if self.n_steps % self.conf.eval_freq == 0:
                self.training = False
                mean_reward, std_reward = self.eval(eval_env)
                self.training = True
                print(
                    f"Mean reward over {self.conf.n_eval_episodes} evaluation episodes: {mean_reward}"
                )
                print(
                    f"Std reward over {self.conf.n_eval_episodes} evaluation episodes: {std_reward}"
                )
            if self.n_steps % self.conf.optimize_freq == 0 and self.n_steps > self.conf.learning_starts:
                # print("optimizing")
                self.policy_net.toggle_optimization_step(True)
                for _ in range(self.conf.gradient_steps):
                    # Perform one step of the optimization (on the policy network)
                    loss = self._optimize_model()
                    if not loss is None:
                        losses.append(loss.item())
                self.policy_net.toggle_optimization_step(False)

            # Update the target network, copying all weights and biases in DQN
            if self.n_steps % self.conf.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if self.n_steps % self.conf.checkpoint_freq == 0:
                model_name = f"checkpoint_{self.n_steps}"
                torch.save(self.policy_net, f"{self.save_path}/{model_name}.model")

            episode_len += 1
            self.n_steps += 1

            if done:
                # no need to reset env because vec env does so automatically
                episode_count += 1
                episode_rewards.append(episode_reward)
                episode_lens.append(episode_len)
                episode_reward = 0
                if episode_count % 4 == 0:
                    print(f"n_updates: {self.n_updates}")
                    print(f"n_episodes: {episode_count}")
                    print(f"n_steps: {self.n_steps}")
                    print(
                        f"Exploration rate: {self.decay_function(self.n_steps)}"
                    )
                    print(f"Average loss (since last log): {np.mean(losses)}")
                    print(
                        f"Average episode reward: {np.mean(episode_rewards)} +/- {np.std(episode_rewards)}"
                    )
                    print(f"Average episode_len: {np.mean(episode_lens)}")
                    print()
                    episode_rewards = []
                    episode_lens = []
                    losses = []
                episode_len = 0
                self.policy_net.on_episode_over()

    def eval(self, eval_env: VecEnv):
        self.policy_net.eval()
        self.policy_net.on_episode_over()
        episode_count = 0
        rewards = []
        episode_reward = 0
        state = torch.tensor(eval_env.reset(),
                             dtype=torch.float).to(self.conf.device)

        while episode_count < self.conf.n_eval_episodes:
            # n_steps does not matter while evaluating
            action = self._select_action(state, 0)
            state, reward, done, _ = eval_env.step(np.array([action.item()]))
            state = torch.tensor(state, dtype=torch.float).to(self.conf.device)
            episode_reward += reward
            if done:
                rewards.append(episode_reward)
                episode_reward = 0
                episode_count += 1
                self.policy_net.on_episode_over()

        rewards = np.array(rewards)

        self.policy_net.train()

        if self.conf.verbose > 1:
            print(f"Rewards: {rewards}")
        if np.mean(rewards) > self.best_eval_reward:
            self.best_eval_reward = np.mean(rewards)
            model_name = f"model{f'_{self.n_steps}' if not self.conf.only_save_best else ''}"
            torch.save(self.policy_net, f"{self.save_path}/{model_name}.model")
            if self.conf.verbose:
                print("New best model found.")
        return np.mean(rewards), np.std(rewards)

    def _select_action(self, state, n_steps):
        sample = random.random()
        eps_threshold = self.decay_function(n_steps)
        if eps_threshold < 0:
            raise ValueError(
                f"Got epsilon {eps_threshold} eps should be between 0 and 1")
        # always follow the policy if we are not training (ei. we are evaluating)
        if sample > eps_threshold or not self.training:
            #print(f"using policy, training {self.training}")
            with torch.no_grad():
                # t.max(-1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                out = self.policy_net(state.to(self.conf.device))
                return out.argmax(dim=-1).detach().cpu()
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]],
                                dtype=torch.long)

    def _optimize_model(self):
        if self.conf.update_sampling_strategy == self.conf.RANDOM_ORDER_UPDATE:
            # only sample 1 time step, so it is the same as just sampling random steps

            # list af list af transations
            # batch_size x episode length
            batch = self.memory.sample_sequential_steps(self.conf.batch_size,
                                                        1,
                                                        replace=True)
        elif self.conf.update_sampling_strategy == self.conf.SEQUENTIAL_UPDATE_FROM_BEGINNING:
            batch = self.memory.sample_whole_episode(self.conf.batch_size)
        elif self.conf.update_sampling_strategy == self.conf.SEQUENTIAL_UPDATE_RANDOM_START:
            batch = self.memory.sample_sequential_steps(
                self.conf.batch_size, self.conf.n_time_steps_per_sample)
        else:
            raise ValueError(
                f"Configured update sampling strategy was invalid: {self.conf.update_sampling_strategy}"
            )
        if batch == []:
            return

        # find the maximum number of time steps in any one sample in the batch
        max_len = self.max_sample_length(batch)
        #print(f"max_len: {max_len}")
        """
        padded_batch, padding_mask = self.pad_batch( # batch_size x max. episode length
            batch, max_len, self.conf.final_state_value)
        max_len += 5
        padding_mask = padding_mask.to(self.conf.device)
        """

        non_final_mask = torch.zeros((self.conf.batch_size, max_len),
                                     dtype=bool).to(self.conf.device)

        state_batch = torch.zeros((self.conf.batch_size, max_len) +
                                  self.env.observation_space.shape).to(
                                      self.conf.device)

        action_batch = torch.zeros((self.conf.batch_size, max_len, 1),
                                   dtype=torch.int64).to(self.conf.device)

        reward_batch = torch.zeros(
            (self.conf.batch_size, max_len)).to(self.conf.device)

        next_state_batch = torch.zeros_like(state_batch).to(
            self.conf.device).to(self.conf.device)

        next_state_values = torch.zeros(
            (self.conf.batch_size, max_len),
            device=self.conf.device).to(self.conf.device)

        padding_mask = torch.zeros((self.conf.batch_size, max_len, 1),
                                   dtype=bool).to(self.conf.device)

        for idx, sample in enumerate(batch):
            sample = Transition(*zip(*sample))  # n_time_steps in each sample
            padding_length = max_len - len(sample.state)

            state_padding = tuple(
                0 for _ in range(len(sample.state[0].shape) * 2 -
                                 1)) + (padding_length, )
            state_batch[idx] = F.pad(torch.cat(sample.state),
                                     state_padding,
                                     value=self.conf.final_state_value)

            next_state_batch[idx] = F.pad(torch.cat(sample.next_state),
                                          state_padding,
                                          value=self.conf.final_state_value)

            reward_padding = tuple(
                0 for _ in range(len(sample.reward[0].shape) * 2 -
                                 1)) + (padding_length, )
            reward_batch[idx] = F.pad(torch.cat(sample.reward),
                                      reward_padding,
                                      value=self.conf.final_state_value)

            action_padding = tuple(
                0 for _ in range(len(sample.action[0].shape) * 2 -
                                 1)) + (padding_length, )
            action_batch[idx] = F.pad(torch.cat(sample.action),
                                      action_padding,
                                      value=self.conf.final_state_value)

            # print(f"sample.done: {sample.done}")

            done_padding = (0, padding_length)
            non_final_mask[idx] = F.pad(~torch.tensor(sample.done),
                                        done_padding,
                                        value=False)

            padding_mask[idx, :len(sample.state), :] = True
        """
            print("SHAPES\n#########################\n")
            print(f"state_padding: {state_padding}")
            print(f"sample.state: {torch.cat(sample.state).shape}")
            print(f"state_batch[idx] after padding: {state_batch[idx].shape}")
            print()

            print(f"reward_padding: {reward_padding}")
            print(f"sample.reward: {torch.cat(sample.reward).shape}")
            print(f"reward_batch[idx] after padding: {reward_batch[idx].shape}")
            print()

            print(f"action_padding: {action_padding}")
            print(f"sample.action: {torch.cat(sample.action).shape}")
            print(f"action_batch[idx] after padding: {action_batch[idx].shape}")
            print()

            print(f"done_padding: {done_padding}")
            print(f"sample.done: {torch.tensor(sample.done).shape}")
            print(f"non_final_mask[idx] after padding: {non_final_mask[idx].shape}")
            print()

            if (padding_mask == False).any():
                print(f"Padding mask[idx]: {padding_mask[idx]}")
                print(f"Padding mask[idx] shape: {padding_mask[idx].shape}")
                print()
            print("\n#########################\n")
        """

        #print(f"len padded_batch: {len(padded_batch)}")
        """
        for idx, sample in enumerate(padded_batch):
            # Transpose the sample (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts sample-array of Transitions
            # to Transition of sample-arrays.
            #print(f"len sample: {len(sample)}")
            sample = Transition(*zip(*sample)) # n_time_steps in each sample
            non_final_mask[idx] = torch.tensor(
                tuple(
                    map(lambda done: not done, sample.done)),
                device=self.conf.device,
                dtype=torch.bool,
            ).to(self.conf.device)
            state_batch[idx] = torch.cat(sample.state).to(self.conf.device)
            action_batch[idx] = torch.cat(sample.action).to(self.conf.device)
            reward_batch[idx] = torch.cat(sample.reward).to(self.conf.device)
            next_state_batch[idx] = torch.cat(sample.next_state).to(
                self.conf.device)
        """

        # print(f"state_batch: {state_batch.flatten()}")
        # print(f"next_state_batch: {next_state_batch.flatten()}")

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #out = self.policy_net(state_batch)
        masked_action_batch = action_batch * padding_mask
        #print(padding_mask)
        #print(masked_action_batch)
        #print(action_batch)

        # print(padding_mask.shape)
        # print(masked_action_batch.shape)
        # print(action_batch.shape)
        state_action_values = self.policy_net(state_batch).gather(
            -1, masked_action_batch) * padding_mask
        # print(f"masked_action_batch: {masked_action_batch}")
        # print(f"out: {out}")
        # print(f"state_action_values: {state_action_values}")
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(2)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # print(reward_batch)
        # print(non_final_mask.flatten())

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                next_state_batch).max(-1)[0].detach()[non_final_mask]
            # Compute the expected Q values
            target_q_values = reward_batch + non_final_mask * self.conf.gamma * next_state_values

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        #print(f"state: {state_action_values.flatten()}")
        #print(f"target {target_q_values.flatten()}")
        loss = criterion(state_action_values, target_q_values.unsqueeze(2))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # we specifically don't call on episode over here, because while we are optimizing we don't use the internal hidden state, so no need to reset it here
        self.n_updates += 1
        return loss

    def max_sample_length(self, batch):
        max_len = -1
        for sample in batch:
            if len(sample) > max_len:
                max_len = len(sample)

        return max_len

    def pad_batch(self, batch, length, pad_value):
        """
            Pads the batch, and returns a matching mask so that 
            batch[i].state * mask[i] returns original values, and zero in all the padded values
            WORKS INPLACE!!
        """
        state_pad_value = (torch.zeros(batch[0][0].state.shape) - 1)
        pad_lengths = np.array([length - len(sample) for sample in batch])
        masks = torch.ones((len(batch), length, 1), dtype=torch.bool)
        #if (not pad_lengths == 0).any():
        #batch_copy = deepcopy(batch)
        padding_transition = Transition(
            state_pad_value,
            torch.tensor([[pad_value]]),
            state_pad_value,
            torch.tensor([pad_value]),
            pad_value,
            True,
        )
        for idx, sample in enumerate(batch):
            masks[idx, len(sample):, :] = False
            for _ in range(pad_lengths[idx]):
                sample.append(padding_transition)

        return batch, masks

    def get_decay_linear(self, eps_start, eps_end, end_step):
        """
            Returns a function of step_num that is linear in the interval [0,end_step], after end_step it returns eps_end
        """
        def func(step_num: int) -> float:
            if step_num > end_step:
                return eps_end
            else:
                return eps_start + (step_num) * (eps_end -
                                                 eps_start) / end_step

        return func

    def get_decay_exp(self, eps_start, eps_end, eps_decay):
        """
            Returns a function f where f(0) = 1 and \lim_{x \to \infty} f(x) = eps_end
        """
        def func(step_num: int) -> float:
            eps = eps_end + (eps_start - eps_end) * math.exp(
                -1. * step_num / eps_decay)
            return eps

        return func

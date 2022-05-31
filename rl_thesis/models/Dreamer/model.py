from rl_thesis.models import BaseModel
from rl_thesis.models import utils
from rl_thesis.models.Dreamer import Config
from rl_thesis.dreamer.Agent import Agent
from rl_thesis.dreamer.WorldModel import WorldModel
from rl_thesis.algorithms.utils.replay_memory import ReplayMemory
from rl_thesis.algorithms.random import RandomPolicy
from rl_thesis.dreamer import driver
from rl_thesis.environments import DreamerGymWrapper, OneHotAction
import json
import numpy as np
import torch
import gym
from tqdm import tqdm
import functools


class RandomDreamerPolicy(RandomPolicy):    
    def __call__(self, observation, state=None, episode_start=None, deterministic=False):
        action, state = super().predict(observation, state, episode_start, deterministic)
        return {"action": action}, state


class DreamerReplayMemory(ReplayMemory):
    def push(self, transition, worker=None):
        """Save a transition - in Dreamer compatible dict. using our ReplayMemory backend"""
        # check if transition is first of its episode
        if transition["is_first"]:
            #print(f"current_episode: {self.current_episode}, inc. current_episode: {self.current_episode + 1}")
            self.current_episode += 1
            self.memory[self.current_episode] = []
            self.episode_reward[self.current_episode] = 0

        self.memory[self.current_episode].append(transition)

        self.episode_reward[self.current_episode] += transition["reward"]
        if self.episode_reward[self.current_episode] > self.priority_min_reward:
            self.priority_episodes.add(self.current_episode)
        else:
            # in cases, like mountain car, where reward decreases as time goes on, we have to remove the episode if the reward goes below the threshold
            self.priority_episodes.discard(self.current_episode)

        self.n_transitions += 1

        # Reject oldest episode if at capacity
        if self.n_transitions > self.capacity:
            self.n_transitions -= len(self.memory[self.min_episode])
            del self.memory[self.min_episode]
            del self.episode_reward[self.min_episode]
            self.priority_episodes.discard(self.min_episode)
            self.min_episode += 1


@functools.total_ordering
class StepCounter:
    def __init__(self):
        self.step = 0

    def inc(self):
        self.step += 1

    def __init__(self, initial=0):
        self.step = initial

    def __int__(self):
        return int(self.step)

    def __eq__(self, other):
        return int(self) == other

    def __ne__(self, other):
        return int(self) != other

    def __lt__(self, other):
        return int(self) < other

    def __add__(self, other):
        return int(self) + other


class DreamerModel(BaseModel):
    def __init__(self, conf : Config, env, policy, policy_config):
        super().__init__(conf, env, policy, policy_config)
        self.conf = conf
        self.env = OneHotAction(DreamerGymWrapper(env, obs_key="ram")) # TODO: Set obs_key through some config, depends on environment
        self.log_dir = utils.create_model_monitor_dir(self.conf.log_dir, self.conf.agent.world_model.recurrence_model, env.spec.id, "")
        self.save_dir = utils.create_model_save_dir(self.conf.models_dir, self.conf.agent.world_model.recurrence_model, env.spec.id, "")

        json_obj = self.conf.to_json_serializable_dict()
        json.dump(json_obj, (self.log_dir / "config.json").open("w"), indent=4)
        

    @classmethod
    def load(self, path, env=None, device="cpu"):
        agent = torch.load(path, map_location=device)
        # device is used in agent to move tensors, this overwrites the saved device to the new device
        agent.device = device
        return agent

    def __debug_print_on_step(self, *args, **kwargs):
        print("on_step handler got:")
        print(f"\targs: {args}")
        print(f"\tkwargs: {kwargs}")

    def __debug_print_on_reset(self, *args, **kwargs):
        print("on_reset handler got:")
        print(f"\targs: {args}")
        print(f"\tkwargs: {kwargs}")

    def __next_batch(self, replay_experience, device="cpu"):
        dataset_config = self.conf.dataset

        samples = replay_experience.sample_sequential_steps(
            batch_size=dataset_config.batch_size, 
            n_time_steps=dataset_config.seq_len, 
            replace=dataset_config.replace)

        batch = { }
        for sample in samples: # len(sample) = 5, will happen 10 times
            transitions = { } # collect multiple transitions for a single sub-sequence

            for transition in sample:
                for k, v in transition.items():
                    if not k in transitions:
                        transitions[k] = []
                    transitions[k].append(v)

            for k, v in transitions.items():
                if not k in batch:
                    batch[k] = []
                batch[k].append(np.stack(v))
        batch = {k: torch.tensor(np.stack(v)).to(device) for k, v in batch.items()}
        batch["is_first"][:,0] = True # Ensure the first entry of all is_first arrays is always True, this is also done in the Dreamer code under their common.Replay._sample_sequence and has downstream influence in a large number of places especially the WorldModel and RSSM.
        #print(f"batch shapes: {[(k,v.shape) for k, v in batch.items()]}")
        return batch


    def train(self, eval_env):
        eval_env = OneHotAction(DreamerGymWrapper(eval_env, obs_key="ram"))

        obs_space = self.env.obs_space
        act_space = self.env.act_space
        best_eval_reward = -np.inf
        print(f"[train] act_space: {act_space}")

        metrics, eval_metrics = { }, { }

        self.train_replay = DreamerReplayMemory(**self.conf.replay)
        #eval_replay = DreamerReplayMemory(capacity=self.conf.replay.capacity // 10)

        train_driver = driver.Driver([self.env])
        train_driver.on_step(self.train_replay.push)
        train_driver.on_reset(self.train_replay.push)
        
        eval_driver = driver.Driver([eval_env])
        #eval_driver.on_step(eval_replay.push)
        #eval_driver.on_reset(eval_replay.push)

        if self.conf.prefill:
            print(f"Prefill replay experience with: {self.conf.prefill} steps")
            random_agent = RandomDreamerPolicy(obs_space, act_space["action"])
            train_driver(random_agent, steps=self.conf.prefill, episodes=1)
            eval_driver(random_agent, episodes=1)
            train_driver.reset()
            eval_driver.reset()

        def add_episode_to_metrics(metrics):
            def _add_episode_to_metrics(args):
                if "mean_return" in metrics:
                    metrics["mean_return"].append(np.sum(args["reward"]))
                else:
                    metrics["mean_return"] = [np.sum(args["reward"])]
            return _add_episode_to_metrics

        train_driver.on_episode(add_episode_to_metrics(metrics))
        eval_driver.on_episode(add_episode_to_metrics(eval_metrics))

        hook_metrics = {} if self.conf.debug else None
        agent = Agent(self.conf.agent, obs_space, act_space, self.conf.dataset.batch_size, self.conf.dataset.seq_len, device=self.conf.device, hook_metrics=hook_metrics).to(self.conf.device)
        self.state = None
        batch = self.__next_batch(self.train_replay, device=self.conf.device)
        self.state, _ = agent.learn(batch, self.state)
        for _ in range(self.conf.pretrain):
            batch = self.__next_batch(self.train_replay, device=self.conf.device)
            self.state, _ = agent.learn(batch, self.state)
        step_counter = StepCounter()
        def train_policy(*args):
            with torch.no_grad():
                return agent.policy(*args, mode="explore" if step_counter.step < self.conf.expl_until else "train")

        def eval_policy(*args):
            with torch.no_grad():
                return agent.policy(*args, mode="eval")

        def train_step(transition, worker):
            if step_counter.step % self.conf.train_every == 0:
                for _ in range(self.conf.train_steps):
                    train_batch = self.__next_batch(self.train_replay, device=self.conf.device)
                    self.state, mets = agent.learn(train_batch, self.state)
                    for key, value in mets.items():
                        if key in metrics:
                            metrics[key].append(value)
                        else:
                            metrics[key] = [value]
            if step_counter.step % self.conf.log_every == 0:
                if hook_metrics is not None:
                    metrics.update(**hook_metrics)
                with open(self.log_dir / "train_metrics.csv", "a") as f:
                    if step_counter.step == self.conf.log_every:
                        # this is the first time we are logging, write header to file
                        f.write(",".join(["train_timestep"] + list(metrics.keys())))
                        f.write("\n")
                    f.write(",".join(str(np.nanmean(v)) for v in [[step_counter.step]] + list(metrics.values())))
                    f.write("\n")
                    [metrics[k].clear() for k in metrics.keys()] # reset metrics
        train_driver.on_step(lambda transition, worker: step_counter.inc())
        train_driver.on_step(train_step)
        if self.conf.agent.world_model.recurrence_model.upper() == "TSSM":
            train_driver.on_episode(lambda a: train_driver.reset())
            eval_driver.on_episode(lambda a: eval_driver.reset())

        def eval_done():
            nonlocal best_eval_reward
            metric_avgs = { }

            for key, value in eval_metrics.items():
                avg = np.mean(value)
                print("")
                print(f"Avg. {key} (over {len(value)} episodes): {avg}")
                metric_avgs[key] = avg

            with open(self.log_dir / "eval_metrics.csv", "a") as fp:
                if step_counter.step == 0:
                    fp.write(",".join(["train_timestep"] + list(metric_avgs.keys())))
                    fp.write("\n")

                fp.write(",".join(str(avg) for avg in [step_counter.step] + list(metric_avgs.values())))
                fp.write("\n")

            if metric_avgs["mean_return"] > best_eval_reward:
                model_name = f"{self.conf.agent.world_model.recurrence_model}"
                if not self.conf.only_save_best:
                    model_name += f"_{step_counter.step}.model"
                else:
                    model_name += ".model"
                if not self.conf.debug:
                    torch.save(agent, self.save_dir / model_name)
                best_eval_reward = metric_avgs["mean_return"]
            

            for key in eval_metrics.keys():
                eval_metrics[key].clear()
        eval_driver.on_driver_done(eval_done)
        
        def maybe_checkpoint_model(transition, worker):
            if self.conf.checkpoint_freq and step_counter.step % self.conf.checkpoint_freq == 0:
                model_name = f"checkpoint_{self.conf.agent.world_model.recurrence_model}_{step_counter.step}.model"
                if not self.conf.debug:
                    torch.save(agent, self.save_dir / model_name)
        train_driver.on_step(maybe_checkpoint_model)
        
        progress_bar = tqdm(total=self.conf.total_timesteps)
        while step_counter.step < self.conf.total_timesteps:
            print("Start evaluation")
            agent.eval()
            eval_driver(eval_policy, episodes=self.conf.n_eval_episodes)

            print("Start training")
            agent.train()
            train_driver(train_policy, steps=self.conf.eval_freq)
            progress_bar.update(self.conf.eval_freq)

        self.env.close()
        eval_env.close()


    def eval(self, eval_env):
        pass


if __name__ == "__main__":
    print("Debugging Dreamer Model Start")
    from rl_thesis.config_parsing import overwrite_default_values
    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    config = Config()
    overwrite_default_values("configs/cartpole/dreamer_very_small.json", config)
    #overwrite_default_values("code/configs/mountain_car/dreamer_config.json", config)

    dreamer = DreamerModel(config, env, None, None)

    dreamer.train(eval_env)

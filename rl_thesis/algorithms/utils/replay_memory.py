from collections import namedtuple
import itertools
import numpy as np
from copy import deepcopy, copy

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'step_num', 'done'))


class ReplayMemory:

    def __init__(self,
                 capacity: int,
                 prioritized_replay_factor: float = 0.,
                 priority_min_reward: float = 0.,
                 favor_episode_ends=False):
        """
        Args
            capacity: the maximum number of time steps kept in the replay buffer, when it fills up, all steps from the earliest episode are discarded
            prioritized_replay_factor: Percentage of the time to sample from trajectories where the reward is greater than priority_min_reward
            priority_min_reward: The minimum reward needed for an episode to be considered when sampling with prioritized replay
        """
        self.memory = {}
        self.capacity = capacity
        self.n_transitions = 0
        self.current_episode = -1
        self.min_episode = 0  # Note -1, as we increment current_episode first in push
        # keeps track of rewards incase we want to only sample from episodes with some amount of reward
        self.episode_reward = {}
        # keeps track of episodes with reward greater than priority_min_reward
        self.priority_episodes = set([])
        self.priority_min_reward = priority_min_reward
        self.prioritized_replay_factor = prioritized_replay_factor
        self.favor_episode_ends = favor_episode_ends

    def push(self, state, action, next_state, reward, step_num, done, is_torch=True):
        """Save a transition"""
        # check if transition is first of its episode

        if is_torch:
            state = state.detach().clone()
            action = action.detach().clone()
            next_state = next_state.detach().clone()
            reward = reward.detach().clone()
        else:
            state = copy(state)
            action = copy(action)
            next_state = copy(next_state)
            reward = copy(reward)
        done = copy(done)

        transition = Transition(state, action, next_state, reward, step_num,
                                done)
        #print(f"adding to episode: {self.current_episode}\n\tstep_num: {step_num}")
        if transition.step_num == 0:
            #print(f"current_episode: {self.current_episode}, inc. current_episode: {self.current_episode + 1}")
            self.current_episode += 1
            self.memory[self.current_episode] = []
            self.episode_reward[self.current_episode] = 0

        self.memory[self.current_episode].append(transition)
        self.episode_reward[self.current_episode] += reward
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

    def get_episodes_to_sample_from(self, batch_size, n_time_steps=None):
        """
            Returns a list of episodes to sample from taking prioritized replay into account. The list will contain at least batch_size elements.
        """
        if np.random.sample() < self.prioritized_replay_factor:
            episodes_to_sample_from = [
                k for k in self.priority_episodes 
                if not n_time_steps or len(self.memory[k]) >= n_time_steps
            ]
            if len(episodes_to_sample_from) < batch_size:
                non_priority_episodes = [
                    x for x in self.memory.keys()
                    if x not in self.priority_episodes
                ]
                non_priority_episodes_to_sample_from = np.random.choice(
                    non_priority_episodes,
                    batch_size - len(episodes_to_sample_from))
                episodes_to_sample_from.extend(
                    non_priority_episodes_to_sample_from)
        else:
            episodes_to_sample_from = list(self.memory.keys())

        return episodes_to_sample_from

    def sample_whole_episode(self, batch_size):
        """
            Returns all transitions from a batch_size episodes. 
            May not return a whole episode if it picks the last episode, and it hasn't completed
        """
        if len(self.memory.keys()) < batch_size:
            return []
        episodes_to_sample_from = self.get_episodes_to_sample_from(batch_size)
        batch = []
        sampled_episodes = np.random.choice(episodes_to_sample_from,
                                            size=batch_size,
                                            replace=False)
        for episode_num in sampled_episodes:
            batch.append(self.memory[episode_num])
        return batch

    def sample_sequential_steps(self, batch_size, n_time_steps, replace=False):
        """
            Returns a batch_size samples each containing n_time_steps transitions
            Args
            replace 
                denotes whether steps from the same episode can be sampled multiple times
            favor_episode_ends 
                denotes wether to sample start from [0, episode_len) and then clip to [0, episode_len - n_time_steps] or to sample directly from the latter interval. First one will have a greater chance of the sample containing an episode end 
        """
        if not replace and len(self.memory.keys()) < batch_size:
            return []
        elif replace and self.n_transitions < batch_size:
            return []

        batch = []
        episodes_to_sample_from = self.get_episodes_to_sample_from(batch_size, n_time_steps)
        # print(f"num. episodes in replay experience: {len(self.memory)}")
        # print(f"num. steps in replay experience: {sum([len(e) for e in self.memory.values()])}")
        # print(f"batch_size: {batch_size}")
        # print(f"n_time_steps: {n_time_steps}")
        # print(f"num. episodes to sample from: {len(episodes_to_sample_from)}")
        sampled_episodes = np.random.choice(
            [
                k for k in episodes_to_sample_from
                if len(self.memory[k]) >= n_time_steps
            ],
            size=batch_size,
            replace=replace,
        )
        for episode_num in sampled_episodes:
            episode_len = len(self.memory[episode_num])
            if self.favor_episode_ends:
                start_index = np.random.randint(0, episode_len)
                start_index = np.clip(start_index, 0, episode_len - n_time_steps)
            else:
                start_index = np.random.randint(0, episode_len - n_time_steps + 1)

            end_index = start_index + n_time_steps

            # NOTE: Casting from numpy.int64 to int, for backwards compatability with python3.6
            start_index = int(start_index)
            end_index = int(end_index)

            episode_slice = itertools.islice(
                self.memory[episode_num],
                start_index,
                end_index,
            )
            batch.append(list(episode_slice))
        #return deepcopy(batch)
        return batch

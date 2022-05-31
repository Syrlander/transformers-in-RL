from rl_thesis.environments.utils import setup_environment
from rl_thesis.models.torch_DQN import Model, Config
from rl_thesis.policies.recurrent_cnn import Policy, Config as PolicyConfig
from rl_thesis.algorithms.utils import ReplayMemory, Transition
import torch

class TestTorchDQN():
    def setup_method(self):
        conf = Config()
        policy = Policy
        policy_conf = PolicyConfig()
        env = setup_environment("MountainCar-v0")
        self.dqn = Model(conf, env, policy, policy_config=vars(policy_conf))
        self.replay_memory = ReplayMemory(1000)
        state_shape = (1, 3,3)
        ep_len = 5
        for i in range(10):
            for step_num in range(ep_len):
                tran = Transition(torch.rand(state_shape), torch.randint(0, 4, (1,1)), torch.rand(state_shape), torch.rand((1,1)), step_num, False)

                self.replay_memory.push(tran.state, tran.action, tran.next_state, tran.reward, tran.step_num, tran.done)
            ep_len += 1

    def test_max_len(self):
        batch = self.replay_memory.sample_sequential_steps(3, 5)
        max_len = self.dqn.max_sample_length(batch)
        assert max_len == 5

    def test_pad_to_len(self):
        batch = self.replay_memory.sample_whole_episode(4)
        # batch = self.transitionfy_batch(batch)
        max_len = self.dqn.max_sample_length(batch)

        batch_copy = []
        for sample in batch:
            batch_copy.append(sample.copy())
        padded, mask = self.dqn.pad_batch(batch_copy, max_len, -1)
        
        assert len(mask) == len(batch)
        padded = self.transitionfy_batch(padded)
        batch = self.transitionfy_batch(batch)
        print(f"padded shape {torch.cat(padded[0].state).shape}")
        print(f"mask shape {mask[0].view(-1,1,1).shape}")
        padded_state_sum = torch.tensor([(torch.cat(padded[i].state) * mask[i].view(-1,1,1)).sum() for i in range(len(batch))])
        not_padded_state_sum = torch.tensor([torch.cat(batch[i].state).sum() for i in range(len(batch))])
        print(padded_state_sum)
        print(not_padded_state_sum)
        print(mask)
        print(batch)
        # print(padded)
        abs_diff = torch.abs(padded_state_sum - not_padded_state_sum)
        assert all(abs_diff < 0.00001)


    def transitionfy_batch(self, batch):
        new_batch = []
        for sample in batch:
            new_batch.append(Transition(*zip(*sample)))
        return new_batch
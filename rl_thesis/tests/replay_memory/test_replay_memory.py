import torch
from rl_thesis.algorithms.utils import ReplayMemory

class TestRCNN():
    def setup_method(self):
        self.replay_buffer = ReplayMemory(90)
        for ep in range(10):
            for t in range(10):
                self.replay_buffer.push(torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]), t, torch.Tensor([0]))
    
    # def test_episode_end(self):
    #     ep_end = self.replay_buffer.get_episode_end(0)
    #     assert ep_end == 10

    #     ep_end = self.replay_buffer.get_episode_end(5)
    #     assert ep_end == 60

    #     ep_end = self.replay_buffer.get_episode_end(9)
    #     assert ep_end == 100

    def test_whole_episode(self):
        batch = self.replay_buffer.sample_whole_episode(1)

        assert len(batch) == 1
        assert len(batch[0]) == 10

    def test_episode_slice(self):
        batch = self.replay_buffer.sample_sequential_steps(1, 4)

        assert len(batch) == 1
        assert len(batch[0]) == 4

    def test_multiple_whole_episodes(self):
        batch = self.replay_buffer.sample_whole_episode(3)
        len_checks = [len(sample) == 10 for sample in batch]
        assert len(batch) == 3
        assert all(len_checks)
    
    def test_multiple_episode_slices(self):
        batch = self.replay_buffer.sample_sequential_steps(3, 4)
        len_checks = [len(sample) == 4 for sample in batch]
        assert len(batch) == 3
        assert all(len_checks)
    
    def test_sequential_steps(self):
        batch = self.replay_buffer.sample_sequential_steps(1, 6)
        prev_step_num = batch[0][0].step_num
        for sample in batch[0][1:]:
            assert prev_step_num + 1 == sample.step_num
            prev_step_num = sample.step_num
    
    def test_sequential_steps_from_beginning(self):
        batch = self.replay_buffer.sample_whole_episode(1)
        prev_step_num = batch[0][0].step_num
        assert prev_step_num == 0
        for sample in batch[0][1:]:
            assert prev_step_num + 1 == sample.step_num
            prev_step_num = sample.step_num


        
        

    
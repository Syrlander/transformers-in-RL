import torch
from rl_thesis.policies.recurrent_cnn import Policy as RCNN

class TestRCNN():
    def setup_method(self):
        self.rcnn = RCNN(
            filter_counts = [32, 64, 64],
            strides = [4, 2, 1],
            conv_sizes = [8, 4, 3],
            lstm_hidden_dim = 512,
            output_dim = 4,
            n_input_channels = 3,
            lstm_input_dim = 3136,
            batch_size = 2,
            device="cpu",
            )

    def test_hidden_states_are_caried(self):
        # test assumes that we are optimizing so set the flag
        self.rcnn.eval()
        x = torch.zeros((2, 8, 3, 84, 84))
        y1 = self.rcnn(x)
        y2 = self.rcnn(x)
        # since the hidden state should change for every step, we should not expect y1 == y2 eventhough they are computed from the same input
        assert not (y1 == y2).all()

    def test_reset_hidden_state(self):
        x = torch.zeros((2, 8, 3, 84, 84))
        y1 = self.rcnn(x)
        self.rcnn.on_episode_over()
        y2 = self.rcnn(x)
        # when we reset the hidden state between calls we expect same input to generate same output
        assert (y1 == y2).all()

    def test_hidden_states_do_not_leak_within_batch(self):
        x = torch.zeros((2, 8, 3, 84, 84))
        y = self.rcnn(x)
        
        # since the two samples in the batch are the same we expect the output to be the same
        assert (y[0] == y[1]).all()
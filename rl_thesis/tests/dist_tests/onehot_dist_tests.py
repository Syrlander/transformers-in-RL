import torch
from rl_thesis.dreamer.dists import OneHotDist
class TestOneHotDist:

    def test_mode_no_duplicates(self):
        logits = torch.tensor(
            [
                [
                    [0,0,0,1],
                    [1,1,2,3],
                    [6,2,1,3],
                    [1,3,2,1]
                ]
            ],
            dtype=torch.float
        )
        mode = torch.tensor(
            [
                [
                    [0,0,0,1],
                    [0,0,0,1],
                    [1,0,0,0],
                    [0,1,0,0]
                ]
            ]
        )
        dist = OneHotDist(logits=logits)
        assert (mode == dist.mode()).all()

    def test_mode_duplicates(self):
        logits = torch.tensor(
            [
                [
                    [1,0,0,1],
                    [3,1,2,3],
                    [6,6,1,3],
                    [1,3,3,1]
                ]
            ],
            dtype=torch.float
        )
        mode = torch.tensor(
            [
                [
                    [1,0,0,0],
                    [1,0,0,0],
                    [1,0,0,0],
                    [0,1,0,0]
                ]
            ]
        )
        dist = OneHotDist(logits=logits)
        assert (mode == dist.mode()).all()
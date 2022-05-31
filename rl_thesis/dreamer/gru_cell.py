from torch import nn
from rl_thesis.dreamer import torch_utils
import torch

class GRUCell(nn.Module):
    """
        Differences from the paper:
            * No normalization, because they say that it had no or marginal effect
            * Neither the input nor output is wrapped in a list 

        Output shape is tested and matches the one in original impl.
    """

    def __init__(self,
                 deterministic_state_dim,
                 hidden_state_dim,
                 act="Tanh",
                 update_bias=-1,
                 hook_fn = None,
                 **kwargs) -> None:
        super(GRUCell, self).__init__()
        self._deterministic_state_dim = deterministic_state_dim
        self._act = torch_utils.get_act(act)
        self._update_bias = update_bias
        self._layer = nn.Linear(
            hidden_state_dim + deterministic_state_dim,
            3 * deterministic_state_dim,
            **kwargs,
        )
        self._norm = nn.LayerNorm((3 * deterministic_state_dim,), eps=0.001) # eps tuned to be close to the results of tensorflow - the two implementations differ, making debugging this quite hard
        #self._norm = nn.LayerNorm((3 * deterministic_state_dim,), eps=0.001) # set eps to match that of tensorflow

        if hook_fn:
            self._layer.register_full_backward_hook(lambda m, i, o: hook_fn("layer", m, i, o))

    def forward(self, hidden_state, deter_state):
        """
            If metrics is given hooks will be registered and all gradients will be written to the metrics (in place)
        """
        concat_state = torch.concat((hidden_state, deter_state), -1)
        parts = self._layer(concat_state)
        # print(f"OURS - parts (pre-norm): {parts} (shape: {parts.shape})")
        
        # parts = self._norm(parts) # Uncomment for DEBUG
        
        # print(f"OURS - parts (post-norm): {parts} (shape: {parts.shape})")
        # output size of layer is 3 * deterministic_state_dim, so this will split in 3 equal parts
        # print(f"OURS - parts: {parts} ({parts.shape})")
        reset, cand, update = torch.split(
            parts,
            self._deterministic_state_dim,
            -1,
        )
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * deter_state
        return output, output

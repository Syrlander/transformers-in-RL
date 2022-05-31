from torch import nn
import torch
from rl_thesis.policies.feed_forward import Policy as MLP
from typing import Dict, List, Union
from rl_thesis.dreamer.cnn import TransposedCnn
import numpy as np
from rl_thesis.dreamer import DistLayer, torch_utils


class Decoder(nn.Module):

    def __init__(self,
                 shapes={},
                 cnn_keys=["image"],
                 mlp_keys=[],
                 act="ELU",
                 model_state_size=1524,
                 mlp_layers=[400, 400, 400],
                 mlp_output_dim=400,
                 cnn_depth=48,
                 cnn_kernels=[5, 5, 6, 6],
                 cnn_strides=2,
                 hook_fn = None) -> None:
        """
        Kwargs:
            act: activation function
            cnn_depth: 
            shapes: dict mapping from key to shape of the value with that key in the forward call 
            cnn_keys: keys of the input to forward to feed through the cnn
            mlp_keys: keys of the input to forward to feed through the mlp
        """
        super(Decoder, self).__init__()
        if mlp_keys:
            if hook_fn:
                mlp_hook = hook_fn.add_model_identifier("MLP")
            else:
                mlp_hook = None
            self.mlp = MLP(mlp_layers, model_state_size, mlp_output_dim, act, hook_fn=mlp_hook)

            self.dists = nn.ModuleDict({
                key: DistLayer(mlp_output_dim, shape)
                for key, shape in shapes.items() if key in mlp_keys
            })

        if cnn_keys:
            output_channels = sum(
                [shape[0] for key, shape in shapes.items() if key in cnn_keys])
            self.transposed_cnn = TransposedCnn(
                32 * cnn_depth,
                output_channels,
                cnn_depth,
                cnn_kernels,
                cnn_strides,
                act,
            )
            self.cnn_in_layer = nn.Linear(model_state_size, 32 * cnn_depth)
        self.act_fn = torch_utils.get_act(act)
        self._mlp_keys = mlp_keys
        self._cnn_keys = cnn_keys
        self._cnn_depth = cnn_depth
        self._shapes = shapes

    def forward(self, x: torch.tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        if self._cnn_keys:
            outputs.update(self.cnn_forward(x))
        if self._mlp_keys:
            outputs.update(self.mlp_forward(x))
        return outputs

    def cnn_forward(self, inp: torch.tensor) -> torch.tensor:
        # we have channels first (as opposed to tensorflow impl) so we take index 0
        channels = {k: self._shapes[k][0] for k in self._cnn_keys}
        x = self.cnn_in_layer(inp)
        # again we have channels first
        x = x.view((-1, 32 * self._cnn_depth, 1, 1))
        x = self.transposed_cnn(x)
        x = x.view(inp.shape[:-1] + x.shape[1:])
        print(f"conv output shape: {x.shape}")
        print(channels)
        means = torch.split(x, list(channels.values()), -3)
        dists = {
            key: torch.distributions.Independent(
                torch.distributions.Normal(mean, 1), 3)
            for (key, _), mean in zip(channels.items(), means)
        }
        return dists

    def mlp_forward(self, x: torch.tensor) -> Dict[str, torch.tensor]:
        out = self.act_fn(self.mlp(x))  # Pass last layer through act as well
        return {key: dist(out) for key, dist in self.dists.items()}

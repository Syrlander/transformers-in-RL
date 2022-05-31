from torch import nn
import torch
from rl_thesis.policies.feed_forward import Policy as MLP
from typing import Dict, List, Union
from rl_thesis.dreamer.cnn import CNN
from rl_thesis.dreamer import torch_utils


class Encoder(nn.Module):

    def __init__(self,
                 shapes={},
                 cnn_keys=["image"],
                 mlp_keys=[],
                 act="ELU",
                 mlp_layers=[400, 400, 400],
                 mlp_output_dim=400,
                 cnn_depth=48,
                 cnn_kernels=[4, 4, 4, 4],
                 cnn_strides=2,
                 hook_fn=None) -> None:
        """
        Kwargs:
            act: activation function
            cnn_depth: 
            shapes: dict mapping from key to shape of the value with that key in the forward call 
            cnn_keys: keys of the input to forward to feed through the cnn
            mlp_keys: keys of the input to forward to feed through the mlp
        """
        super(Encoder, self).__init__()

        if mlp_keys:
            input_shape = sum([
                shape[-1] for key, shape in shapes.items() if key in mlp_keys
            ])
            if hook_fn:
                mlp_hook = hook_fn.add_model_identifier("MLP")
            else:
                mlp_hook = None
            self.mlp = MLP(mlp_layers, input_shape, mlp_output_dim, act, hook_fn=mlp_hook)

        if cnn_keys:
            in_channels = sum(
                [shape[0] for key, shape in shapes.items() if key in cnn_keys])
            if hook_fn:
                print("Warning: hook functions are not implemented in CNN encoder")
            self.cnn = CNN(in_channels, cnn_depth, cnn_kernels, cnn_strides,
                           act)

        self._mlp_keys = mlp_keys
        self._cnn_keys = cnn_keys
        self.act = torch_utils.get_act(act)

    def forward(self, x: Dict[str, torch.tensor]) -> torch.tensor:
        outputs = []
        if self._cnn_keys:
            # Value shapes: (batch_size, n_time_steps, channels, width, height)
            inputs = [
                value for key, value in x.items() if key in self._cnn_keys
            ]

            # concat on dim. 2 (channels) if we have more than one modality of input (e.g. screen and heatmap)
            outputs.append(self.cnn_forward(torch.concat(inputs, 2)))
        if self._mlp_keys:
            # values shapes: (batch_size, n_time_steps, input_size)
            inputs = [
                value for key, value in x.items() if key in self._mlp_keys
            ]
            outputs.append(self.mlp_forward(torch.concat(inputs, -1)))

        output = torch.concat(outputs, -1)
        return output

    def cnn_forward(self, x: torch.tensor) -> torch.tensor:
        # Reshape from (batch_size, n_time_steps, channels, width, height) to (batch_size * n_time_steps, channels, width, height)
        batch_size, n_timesteps = x.shape[:2]
        x = x.view((batch_size * n_timesteps, ) + x.shape[2:])
        out = self.act(self.cnn(x))

        return out.view(batch_size, n_timesteps, -1)  # Flatten last dim.

    def mlp_forward(self, x: torch.tensor) -> torch.tensor:
        return self.act(self.mlp(x))

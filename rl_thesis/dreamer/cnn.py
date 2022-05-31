from torch import nn
import torch
from typing import Dict, List, Union


class CNN(nn.Module):

    def __init__(
        self,
        input_channel_count: int,
        depth: int,
        kernels: List[int],
        strides: Union[List[int], int],
        act_fn: str,
    ) -> None:
        """

        Args:
            input_channel_count: number of channels of input
            depth: int for depth to expand convolutions by at each layer
            kernels: list of kernel sizes to use at each layer
            strides: list of strides to use at each layer, or if int use the same stride at each layer
            act_fn: name of activation function to use
        """
        super().__init__()
        self._input_channel_count = input_channel_count
        self._depth = depth
        self._kernels = kernels

        if isinstance(strides, int):
            self._strides = [strides] * len(kernels)
        elif isinstance(strides, list):
            self._strides = strides
        else:
            raise ValueError(
                f"Got invliad value for argument 'strides'. Got value: {strides}"
            )

        self._act = getattr(nn, act_fn)()

        self._depths = [input_channel_count
                        ] + [2**i * depth for i in range(len(kernels))]
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=self._depths[i],
                      out_channels=self._depths[i + 1],
                      kernel_size=kernels[i],
                      stride=self._strides[i]) for i in range(len(kernels))
        ])

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x expected shape: (batch_size  n_time_steps, channels, width, height)
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
        return self.layers[-1](x)


class TransposedCnn(nn.Module):

    def __init__(
        self,
        input_channel_count: int,
        output_channel_count: int,
        depth: int,
        kernels: List[int],
        strides: Union[List[int], int],
        act_fn: str,
    ) -> None:
        super(TransposedCnn, self).__init__()
        self._output_channel_count = output_channel_count
        self._depth = depth
        self._kernels = kernels

        if isinstance(strides, int):
            self._strides = [strides] * len(kernels)
        elif isinstance(strides, list):
            self._strides = strides
        else:
            raise ValueError(
                f"Got invliad value for argument 'strides'. Got value: {strides}"
            )

        self._act = getattr(nn, act_fn)()

        self._depths = [input_channel_count] + [
            2**(len(kernels) - i - 2) * depth for i in range(len(kernels) - 1)
        ] + [output_channel_count]
        print(f'depths {self._depths}')
        self.layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=self._depths[i],
                               out_channels=self._depths[i + 1],
                               kernel_size=kernels[i],
                               stride=self._strides[i])
            for i in range(len(self._depths) - 1)
        ])

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x expected shape: (batch_size  n_time_steps, channels, width, height)
        print(f"Transposed conv input x shape {x.shape}")
        for layer in self.layers[:-1]:
            x = self._act(layer(x))
            print(f"Transposed conv x shape {x.shape}")
        return self.layers[-1](x)
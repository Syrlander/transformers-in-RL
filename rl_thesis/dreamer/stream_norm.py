import torch
from torch import nn


class StreamNorm():
    """
        Normalizes batch of inputs either based only on current batch, or on mix of earlier batches depending on the momentum.
    """

    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        # Momentum of 0 normalizes only based on the current batch.
        # Momentum of 1 disables normalization.
        self._shape = tuple(shape)
        self._momentum = momentum
        self._scale = scale
        self._eps = eps
        self.mag = torch.ones(shape)

    def __call__(self, inputs):
        metrics = {}
        self.update(inputs)
        metrics['mean'] = inputs.mean()
        metrics['std'] = inputs.std()
        outputs = self.transform(inputs)
        metrics['normed_mean'] = outputs.mean()
        metrics['normed_std'] = outputs.std()
        return outputs, metrics

    def reset(self):
        self.mag = torch.ones_like(self.mag)

    def update(self, inputs):
        batch = inputs.reshape((-1, ) + self._shape)
        mag = torch.abs(batch).mean(0)
        self.mag = self._momentum * self.mag + (1 - self._momentum) * mag

    def transform(self, inputs):
        values = inputs.reshape((-1, ) + self._shape)
        values /= self.mag[None] + self._eps
        values *= self._scale
        return values.view(inputs.shape)
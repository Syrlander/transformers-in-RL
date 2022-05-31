import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from rl_thesis.utils.hooks import BackwardHookFunction
from rl_thesis.policies.feed_forward import Policy as MLP
class MLPWithDist(MLP):

    def __init__(self, hidden_layer_sizes, input_dim, output_dim, act_fn,
                 *dist_args, hook_fn: BackwardHookFunction = None, **dist_kwargs):
        
        last_hidden_dim = hidden_layer_sizes[-1]

        if hook_fn:
            mlp_hook = hook_fn.add_model_identifier("MLP")
            dist_hook = hook_fn.add_model_identifier("DistLayer")
        else:
            mlp_hook = None
            dist_hook = None
        super().__init__(hidden_layer_sizes[:-1], input_dim, last_hidden_dim,
                         act_fn, hook_fn=mlp_hook)
        self.dist = DistLayer(last_hidden_dim, output_dim, *dist_args,
                              **dist_kwargs, hook_fn=dist_hook)

    def forward(self, x: torch.tensor):
        out = self.act(super().forward(x))
        return self.dist(out)

class DistLayer(nn.Module):

    def __init__(self,
                 inp_dim,
                 out_shape,
                 dist: str = "mse",
                 min_std: float = 0.1,
                 init_std: float = 0.0,
                 hook_fn : BackwardHookFunction = None) -> None:
        super().__init__()
        self._inp_dim = inp_dim
        
        self._out_shape = out_shape

        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std

        self.out_layer = nn.Linear(inp_dim, np.prod(out_shape, dtype=int))
        self.std_layer = nn.Linear(inp_dim, np.prod(out_shape, dtype=int))
        if hook_fn:
            self.out_layer.register_full_backward_hook(lambda m, i, o: hook_fn("out_layer", m, i, o))
            self.std_layer.register_full_backward_hook(lambda m, i, o: hook_fn("std_layer", m, i, o))

    def forward(self, inputs: torch.tensor):
        out = self.out_layer(inputs)
        out = out.view(inputs.shape[:-1] + tuple(self._out_shape))

        if self._dist in ("normal", "tanh_normal", "trunc_normal"):
            std = self.std_layer(inputs)
            std = std.view(inputs.shape[:-1] + tuple(self._out_shape))

        if self._dist == "mse":
            # print(f"OURS - out: {out} (shape: {out.shape})")
            # print(f"OURS: out_shape: {self._out_shape} (len: {len(self._out_shape)})")
            dist = torch.distributions.Normal(out, 1)
            return torch.distributions.Independent(dist, len(self._out_shape))
            #tmp = torch.distributions.Independent(dist, len(self._out_shape))
            # print(f"OURS - mean post construct: {tmp.mean}")
            #return tmp
        if self._dist == "normal":
            dist = torch.distributions.Normal(out, std)
            return torch.distributions.Independent(dist, len(self._out_shape))
        if self._dist == "binary":

            return Bernoulli(
                torch.distributions.independent.Independent(
                    torch.distributions.bernoulli.Bernoulli(logits=out), 
                    len(self._out_shape)
                    )
                )
        if self._dist == "tanh_normal":
            mean = 5 * F.tanh(out / 5)
            std = F.softplus(std + self._init_std) + self._min_std
            dist = torch.distributions.Normal(mean, std)
            dist = torch.distributions.TransformedDistribution(
                dist,
                torch.distributions.transforms.TanhTransform(cache_size=1))
            return torch.distributions.Independent(dist, len(self._out_shape))
        if self._dist == "trunc_normal":
            std = 2 * nn.Sigmoid((std + self._init_std) / 2) + self._min_std
            dist = TruncatedNormalDist(torch.tanh(out), std, -1, 1)
            return torch.distributions.Independent(dist, 1)
        if self._dist == "onehot":
            # print(f"out: {out}")
            return OneHotDist(logits=out)
        raise NotImplementedError(
            f"Got invalid distribution layer name: '{self._dist}'")


#class OneHotDist(torch.distributions.OneHotCategoricalStraightThrough):
class OneHotDist(torch.distributions.OneHotCategorical):

    #def sample(self, *args, **kwargs):
    def sample(self, sample_shape=torch.Size()):
        # Straight through gradients
        # also see: https://github.com/pytorch/pytorch/blob/0aa3c39e5f296dd0871d0f849e295d3b7644ff2e/torch/distributions/one_hot_categorical.py#L105-L118
        #return super().rsample(*args, **kwargs)
        #
        # NOTE: Code taken from OneHotCategoricalStraightThrough, but it calls self.sample, instead of super().sample so as we need a sample method the on in OneHotCategoricalStraightThrough will end up calling our version, hence we reach the recursion depth limit.
        samples = super().sample(sample_shape)
        probs = self._categorical.probs
        return samples + (probs - probs.detach())

    def mode(self):
        mode = torch.zeros_like(self.logits)
        mode[self.logits == self.logits.max(dim=-1, keepdim=True)[0]] = 1
        
        if not (mode.sum(dim=-1) == 1).all():
            greater_than_one = mode.sum(dim=-1) > 1
            rows_to_change = mode[greater_than_one]
            for i in range(len(rows_to_change)):
                argmax = rows_to_change[i].argmax()
                new_row = torch.zeros_like(rows_to_change[i])
                new_row[argmax] = 1
                rows_to_change[i] = new_row
            mode[greater_than_one] = rows_to_change

        return mode.detach() + (super().logits - super().logits.detach()) # NOTE: Similar to jsikyoon implementation under tools.py: OneHotDist, also includes the straight through gradients in the mode


class TruncatedNormalDist(torch.distributions.normal.Normal):

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1) -> None:
        super().__init__(loc, scale)
        self._clip = clip
        self._mult = mult
        self._low = low
        self._high = high

    def sample(self, *args, **kwargs):
        event = super().sample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(event,
                                  min=self._low + self._clip,
                                  max=self._high - self._clip)
            # keep only the clipped values, but with gradients of event
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event

class Bernoulli:
    """
        Bernouli distribution taken from https://github.com/jsikyoon/dreamer-torch/blob/main/tools.py
    """
    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() +self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1-x) + log_probs1 * x
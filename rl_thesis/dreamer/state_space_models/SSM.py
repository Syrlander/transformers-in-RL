import numpy as np
import torch
from torch import nn
from typing import Tuple, List
from rl_thesis.dreamer import torch_utils
import torch.nn.functional as F
from rl_thesis.dreamer import StateData
from rl_thesis.dreamer.dists import DistLayer, OneHotDist, MLPWithDist
from rl_thesis.dreamer import GRUCell

class SSM(nn.Module):

    def __init__(self,
                 seq_len: int,
                 batch_size: int,
                 discrete: int = 32,
                 stoch_size: int = 32,
                 model_state_size: int = 1536,
                 hidden_state_dim: int = 200,
                 deterministic_state_dim: int = 200,
                 act_fn="ELU",
                 std_act="softplus",
                 min_std=0.1,
                 ensemble=5,
                 action_space_size=4,
                 hook_fn=None,
                 transformer_config=None) -> None:
        super(SSM, self).__init__()

        self._discrete = discrete
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._stoch_size = stoch_size
        self._ensemble = ensemble
        self._deterministic_state_dim = deterministic_state_dim

        self.distribution_construction_layers = nn.ModuleDict({
            "img_in":
            nn.Linear(stoch_size * self._discrete + action_space_size,
                      hidden_state_dim)
        })

        if self._discrete:
            self.distribution_construction_layers.update({
                "obs_dist":
                nn.Linear(hidden_state_dim, self._discrete * self._stoch_size)
            })
        else:
            self.distribution_construction_layers.update({
                "obs_dist":
                nn.Linear(hidden_state_dim, 2 * self._stoch_size)
            })
        self.distribution_construction_layers.update({
            f"img_dist_{k}": nn.Linear(hidden_state_dim,
                                       self._discrete * self._stoch_size)
            for k in range(ensemble)
        })
        self.distribution_construction_layers.update({
            f"img_out_{k}": nn.Linear(self._deterministic_state_dim,
                                      hidden_state_dim)
            for k in range(ensemble)
        })
        if hook_fn:
            self.observation_out_layer.register_full_backward_hook(lambda m, i, o: hook_fn("obs_out", m, i, o))
            for name, layer in self.distribution_construction_layers.items():
                layer.register_full_backward_hook(lambda m, i, o: hook_fn(name, m, i, o))
        # activation function to apply to the standard deviation in continuous case
        self._std_act = std_act
        self._min_std = min_std

        self.act_fn = torch_utils.get_act(act_fn)

    def initialize_state(self, batch_size=None):
        batch_size = self._batch_size if batch_size is None else batch_size
        return StateData.get_initial_state(
            self._discrete,
            batch_size,
            self._stoch_size,
            self._deterministic_state_dim,
            device=next(self.parameters()).device
        )

    def observe(self, embed, prev_actions, is_first, prev_state=None):
        """


        We wish to loop over each timestep of each batch sample and feed them
        through. Hence, we iterate over sequence length and NOT batch size.

        Input shapes/types:
            embed: (batch size x n timesteps x embed dim.)
            prev_actions: (batch size x n timesteps x action space dim.)
            is_first: (batch size x n timesteps)
            prev_state: instance of StateData. Initilize everything in StateData to zeros if None

        Args:
            embed: 
            action: 
            is_first: 
        Kwargs:
            state: 
        """
        pass

    def compute_stochastic_posterior(self, x, sample=True):
        x = self.observation_out_layer(x)
        x = self.act_fn(x)

        distribution_description = self.construct_dist_stats("obs_dist", x)
        distribution = self.distribution_from_stats(distribution_description)
        stoch = distribution.sample() if sample else distribution.base_dist.mode()  # stochastic part of the state
        return stoch, distribution_description

    def construct_dist_stats(self, name, x) -> StateData:
        """
        Corresponds to _suff_stats_layer
        """
        distribution_construction_layer = self.distribution_construction_layers[
            name]
        if self._discrete:
            x = distribution_construction_layer(x)
            logit = x.view(
                list(x.shape[:-1]) + [self._stoch_size, self._discrete])
            return StateData(logit=logit)
        else:
            x = distribution_construction_layer(x)
            mean, std = torch.split(x, self._stoch_size, -1)
            # The assignment to std basically works like a switch-statement
            # Only the indexed value is ever evaluated!
            std = {
                'softplus': lambda: nn.Softplus()(std),
                'sigmoid': lambda: nn.Sigmoid()(std),
                'sigmoid2': lambda: 2 * nn.Sigmoid()(std / 2),
            }[self._std_act]()
            std += self._min_std
            return StateData(mean=mean, std=std)

    def construct_dist_stats_ensemble(self, inp) -> StateData:
        """
        Corresponds to _suff_stats_ensemble

        TODO: Maybe remove this (and StateData.stack), since ensemble doesn't even seem to be used
              Just use construct_dist_stats_layer instead and ignore all ensemble stuff.
        """
        # TODO: Maybe reshape
        # batch_size = list(inp.shape[:-1])

        stats = []  # List of StateData
        for k in range(self._ensemble):
            x = self.distribution_construction_layers[f"img_out_{k}"](inp)
            x = self.act_fn(x)
            stats.append(self.construct_dist_stats(f"img_dist_{k}", x))

        stats = StateData.stack(stats)

        return stats

    def distribution_from_stats(self, state: StateData):
        """
        Corresponds to get_dist

        Args.
            state: 

        Remarks:
            ensemble argument has been removed, since the original code never
            calls get_dist with ensemble set to anything other than False
        """
        if self._discrete:
            # Cast to float32 because autocast will change the type and then stuff breaks
            logits = state.logit.type(torch.float32)
            try:
                return torch.distributions.Independent(OneHotDist(logits=logits), 1)
            except ValueError as e:
                print("Failed")
                print(f"is inf: {torch.isinf(logits).any()}")
                print(f"is nan: {torch.isnan(logits).any()}")
                print(f"type: {logits.type()}")
                raise e
        else:
            mean, std = state.mean, state.std
            return torch.distributions.MultivariateNormal(mean, std)

    def get_feat(self, state):
        """
        state:
            * deter: [16, 50, 600]
            * logit: [16, 50, 32, 32]
            * stoch: [16, 50, 32, 32]
        """
        stoch = state.stoch  # One hot vectors like the ones on page 3, Figure 2
        if self._discrete:
            shape = stoch.shape[:-2] + (self._stoch_size * self._discrete, )
            stoch = stoch.view(shape)
        return torch.concat([stoch, state.deter], -1)

    def imagine(self, action, state=None):
        if state is None:
            state = self.initialize_state()
        prior = state
        priors = []
        for t in range(self._seq_len):
            prior = self.imagine_step(prior, action[:, t])
            priors.append(prior)

        return StateData.stack(priors, dim=1)

    def imagine_step(self, prev_state, prev_action, sample=True) -> StateData:
        pass
    
    def kl_loss(self, post: StateData, prior: StateData, forward: bool,
                balance: float, free: float, free_avg: bool):
        """
        balance: 0.8
        forward: False
        free: 0.0
        free_avg: True
        post:
            deter: [16, 50, 600]
            logit: [16, 50, 32, 32]
            stoch: [16, 50, 32, 32]
        prior:
            deter: [16, 50, 600]
            logit: [16, 50, 32, 32]
            stoch: [16, 50, 32, 32]
        """
        sg = lambda d: StateData(**{k: v.detach() for k, v in d.to_dict().items()})

        free = torch.tensor(free)

        lhs, rhs = (prior, post) if forward else (post, prior)
        mix = balance if forward else (1 - balance)
        if balance == 0.5:
            value = F.kl_div(self.distribution_from_stats(lhs),
                             self.distribution_from_stats(rhs))
            loss = torch.maximum(value, free).mean()
        else:
            value_lhs = value = torch.distributions.kl_divergence(
                self.distribution_from_stats(lhs),
                self.distribution_from_stats(sg(rhs)))
            value_rhs = torch.distributions.kl_divergence(
                self.distribution_from_stats(sg(lhs)),
                self.distribution_from_stats(rhs))

            #value_lhs = value = torch.distributions.kl_divergence(
            #    self.distribution_from_stats(lhs),
            #    with_no_grad(self.distribution_from_stats, rhs))
            #value_rhs = torch.distributions.kl_divergence(
            #    with_no_grad(self.distribution_from_stats, lhs),
            #    self.distribution_from_stats(rhs))
            if free_avg:
                loss_lhs = torch.maximum(value_lhs.mean(), free)
                loss_rhs = torch.maximum(value_rhs.mean(), free)
            else:
                loss_lhs = torch.maximum(value_lhs, free).mean()
                loss_rhs = torch.maximum(value_rhs, free).mean()
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss, value


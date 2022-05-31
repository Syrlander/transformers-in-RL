import numpy as np
import torch
from torch import nn
from typing import Tuple, List
from rl_thesis.dreamer import torch_utils
import torch.nn.functional as F
from rl_thesis.dreamer import StateData
from rl_thesis.dreamer.dists import DistLayer, OneHotDist, MLPWithDist
from rl_thesis.dreamer import GRUCell
from .SSM import SSM

class RSSM(SSM):

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
        super().__init__(seq_len,
                         batch_size,
                         discrete,
                         stoch_size,
                         model_state_size,
                         hidden_state_dim,
                         deterministic_state_dim,
                         act_fn,
                         std_act,
                         min_std,
                         ensemble,
                         action_space_size,
                         hook_fn,)

        # Layer is defined here because in TSSM the posterior is not dependend on the deterministic state, so the input shapes will differ
        self.observation_out_layer = nn.Linear(
            model_state_size + self._deterministic_state_dim, hidden_state_dim)

        gru_hook_fn = hook_fn.add_model_identifier("GRU") if hook_fn else None
        self._cell = GRUCell(deterministic_state_dim, hidden_state_dim, hook_fn=gru_hook_fn)

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
        if prev_state is None:
            prev_state = self.initialize_state()

        priors = []
        posts = []

        #print(f"is_not_first: {is_not_first}")
        for t in range(self._seq_len):
            #print(f"prev_actions[:, t]: {prev_actions[:, t]}")
            
            #print(f"masked prev action: {prev_action}")
            #print(f"prev state: {prev_state}")
            #print(f"is_not_first[:, t] shape: {is_not_first[:, t].shape}")

            #print(f"is not first: {is_not_first}")
            #print(f"masked_prev_state: {masked_prev_state}")
            post, prior = self.observation_step(embed[:, t], prev_actions[:,t],prev_state, is_first[:,t])

            priors.append(prior)
            posts.append(post)

            prev_state = post

        priors = StateData.stack(priors, dim=1)
        posts = StateData.stack(posts, dim=1)
        return priors, posts

    def observation_step(self,
                         embed,
                         prev_action,
                         prev_state,
                         is_first,
                         sample=True) -> Tuple[StateData, StateData]:
        """
            embed: [16, 400]
            prev_action: [16, 2]
            prev_state:
                * deter: [16, 600]
                * logit: [16, 32, 32]
                * stoch: [16, 32, 32]
            is_first: [16] (dtype: bool)

            Prev action and state has been masked by is_first in RSSM.observe, so no need to here 
        """

        #is_not_first = (~is_first).unsqueeze(-1)
        is_not_first = (1.0 - is_first.type(torch.float32))

        # print(f"OURS - is_not_first shape: {is_not_first.shape}")

        masked_prev_state = StateData(
            **{
                k: torch.einsum("b,b...->b...", is_not_first, v)  # multiply elementwise along first dimension
                for k, v in prev_state.to_dict().items()
            })

        # print(f"OURS - masked_prev_state.logit: {masked_prev_state.logit}")
        # print(f"OURS - masked_prev_state.deter: {masked_prev_state.deter}")
        # print(f"OURS - masked_prev_state.stoch: {masked_prev_state.stoch}")

        prev_action = prev_action * is_not_first.unsqueeze(-1)
        # print(f"OURS - prev_action: {prev_action}")

        prior = self.imagine_step(masked_prev_state, prev_action, sample)
        # prior = prev_state # NOTE: Uncomment this line for debugging, since imagine_step uses the LayerNorm of the GRU, causing a difference between the two implementations

        # print(f"OURS - prior.logit: {prior.logit}")

        x = torch.concat([prior.deter, embed], -1)
        stoch, distribution_description = self.compute_stochastic_posterior(x, sample=sample)
        post = StateData(stoch=stoch,
                         deter=prior.deter,
                         **distribution_description.to_dict())
        return post, prior

    def imagine_step(self, prev_state, prev_action, sample=True) -> StateData:
        """
            If metrics is given it will be passed to the cell's forward call

            prev_state: StateData:
                * deter: [16, 1024]
                * logit: [16, 32, 32]
                * stoch: [16, 32, 32]
            prev_action: [16, 2] (all zeros, or OneHotEncoded)
        """
        prev_stoch = prev_state.stoch
        if self._discrete:
            prev_stoch = torch.flatten(prev_stoch, start_dim=-2)
        x = torch.concat([prev_stoch, prev_action], -1)
        #print(f"x1 (ours): {x} ({x.shape})")
        l = self.distribution_construction_layers["img_in"]
        x = l(x)
        #print(l)
        #print(f"OURS l: {l}")
        #print("OUR l weight")
        #print(l.weight.data)
        #print("OUR l bias")
        #print(l.bias.data)
        #print(dir(l))
        #print(f"x2 (ours): {x} ({x.shape})")
        x = self.act_fn(x)

        #print(f"x3 (ours): {x} ({x.shape})")
        deter = prev_state.deter
        #print(f"start deter (ours): {deter} ({deter.shape})")
        x, deter = self._cell(x, deter)


        #print(f"x4 (ours): {x} ({x.shape})")
        #print(f"OURS - deter: {deter} ({deter.shape})")
        stats = self.construct_dist_stats_ensemble(x)
        # NOTE: No random sampling for ensembling, since they use ensemble = 1 anyway and don't mentioned it in their paper
        stats = StateData.unstack(stats)[0]  # Unpack first 'stack' dim.
        # print(f"stats['logit'] (ours): {stats.logit}")
        dist = self.distribution_from_stats(stats)
        #stoch = dist.base_dist.mode() # DEBUG
        stoch = dist.sample() if sample else dist.mode() # TODO: This case will fail, since pytorch dists don't have mode
        prior = StateData(stoch=stoch, deter=deter, **stats.to_dict())
        return prior
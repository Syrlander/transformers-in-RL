from rl_thesis.dreamer import state_space_models as ssm
from rl_thesis.dreamer import StateData
from torch import nn
import torch
from typing import Tuple
from rl_thesis.gated_transformer.mem_transformer import PositionalEmbedding
class TSSM(ssm.SSM):

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
        super().__init__(seq_len, batch_size, discrete, stoch_size, model_state_size, hidden_state_dim, deterministic_state_dim, act_fn, std_act, min_std, ensemble, action_space_size, hook_fn)

        self.observation_out_layer = nn.Linear(model_state_size, hidden_state_dim)
        self._cell = nn.Transformer(
            d_model=hidden_state_dim,
            batch_first=True,
            **transformer_config
        )
        self.positional_encoder = PositionalEmbedding(hidden_state_dim)
        

    def observe(self, embeds, prev_actions, is_first, prev_state=None):
        sample = True
        if prev_state is None:
            prev_state = self.initialize_state()

        posts, post_stats = self.compute_stochastic_posterior(embeds, sample=sample)
        deters = self.compute_deters(posts, prev_actions)
        priors, prior_stats = self.compute_priors(deters, prev_actions, sample=sample)
        posts = StateData(stoch=posts, deter=deters, **post_stats.to_dict())
        priors = StateData(stoch=priors, deter=deters, **prior_stats.to_dict())
        return priors, posts

    def compute_deters(self, posts, actions):
        if self._discrete:
            posts_for_transformer = torch.flatten(posts, start_dim=-2)
        else:
            posts_for_transformer = posts
        x = torch.concat([posts_for_transformer, actions], -1)
        layer = self.distribution_construction_layers["img_in"]
        x = layer(x)
        x = self.act_fn(x)

        deters = self.transformer(x)

        return deters

    def transformer(self, x):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        device = next(self.parameters()).device
        # Masks that make sure that none of the attention mechanisms can attend to future states
        transformer_masks = {
            "tgt_mask": nn.Transformer.generate_square_subsequent_mask(seq_len).to(device),
            "src_mask": nn.Transformer.generate_square_subsequent_mask(seq_len + 1).to(device),
            "memory_mask": self.generate_memory_mask(seq_len + 1, seq_len).to(device),
        }
        src = torch.concat([self.get_start_of_sequence(x), x], dim=1) 
        # add one because we just added the SOS token
        src_embedding = self.positional_encoder(torch.arange(0, seq_len + 1, device=device), bsz=batch_size).transpose(0, 1)
        tgt_embedding = self.positional_encoder(torch.arange(0, seq_len, device=device), bsz=batch_size).transpose(0,1)
        

        out = self._cell(src + src_embedding, x + tgt_embedding, **transformer_masks)
        return out

    def observation_step(self,
                         embed,
                         prev_actions,
                         prev_states : StateData, # prev posterior states
                         is_first,
                         sample=True) -> Tuple[StateData, StateData]:
        
        """
            First iteration expected lengths:
                embed: 1
                prev actions: 1
                prev states: 0
        """

        stoch_post, post_stats = self.compute_stochastic_posterior(embed, sample=sample)
        
        # TODO: Check shapes to see if this is correct dim (I don't think it is)
        prev_post_stochs = torch.concat([prev_states.stoch, stoch_post.unsqueeze(0)], dim=0)
        # TODO: this should compute all the deters, or at least computation of the next state should have attention to the all previous states
        deter = self.compute_deters(prev_post_stochs, prev_actions)
        post = StateData(stoch=stoch_post, deter=deter[-1], **post_stats.to_dict())

        def atleast_nd(tensor, n):
            if len(tensor.shape) >= n:
                return tensor
            else:
                ones = tuple((1 for _ in range(n - len(tensor.shape))))
                new_shape = ones + tensor.shape
                return tensor.view(new_shape)

        def stack_func(tensors, dim=0):
            n_dims = torch.tensor([len(t.shape) for t in tensors])
            max_dims = n_dims.max()
            if (n_dims > 4).any() or (n_dims < 2).any():
                print("Funky shape, don't think this works as expected")
                print(n_dims)
            new_tensors = [atleast_nd(t, max_dims) for t in tensors]
            return torch.concat(new_tensors, dim=dim)

            

        posts = StateData.stack([prev_states, post], stack_func=stack_func)
        return posts, None

    def initialize_state(self, batch_size=None):
        batch_size = self._batch_size if batch_size is None else batch_size
        state = StateData.get_initial_state(
            self._discrete,
            batch_size,
            self._stoch_size,
            self._deterministic_state_dim,
            device=next(self.parameters()).device,
            seq_len=0
            )
        return state

    def get_start_of_sequence(self, sequence):
        """
            sequence.shape = (batch_size, seq_len, embed_dim)
        """
        batch_size, _, embed_dim = sequence.shape
        return torch.zeros((batch_size, 1, embed_dim), device=next(self.parameters()).device)

    def generate_memory_mask(self, src_seq_len, tgt_seq_len):
        memory_mask = torch.zeros(tgt_seq_len, src_seq_len)
        for i in range(memory_mask.shape[0]):
            for j in range(memory_mask.shape[1]):
                if j > i:
                    memory_mask[i,j] = float("-inf")
        return memory_mask

    def compute_priors(self, states, actions, sample=True):
        stats = self.construct_dist_stats_ensemble(states)
        # NOTE: No random sampling for ensembling, since they use ensemble = 1 anyway and don't mentioned it in their paper
        stats = StateData.unstack(stats)[0]  # Unpack first 'stack' dim.
        dist = self.distribution_from_stats(stats)
        stoch = dist.sample() if sample else dist.base_dist.mode()
        return stoch, stats
 
    def imagine_step(self, prev_states, actions, sample=True) -> StateData:
        """
            If metrics is given it will be passed to the cell's forward call

            prev_state: StateData:
                * deter: [16, 1024]
                * logit: [16, 32, 32]
                * stoch: [16, 32, 32]
            prev_action: [16, 2] (all zeros, or OneHotEncoded)
        """
        prev_stoch = prev_states.stoch
        if self._discrete:
            prev_stoch = torch.flatten(prev_stoch, start_dim=-2)
        x = torch.concat([prev_stoch, actions], -1)
        l = self.distribution_construction_layers["img_in"]
        x = l(x)
        x = self.act_fn(x)
        
        deters = self.transformer(x)


        stats = self.construct_dist_stats_ensemble(deters[-1])
        # NOTE: No random sampling for ensembling, since they use ensemble = 1 anyway and don't mentioned it in their paper
        stats = StateData.unstack(stats)[0]  # Unpack first 'stack' dim.
        dist = self.distribution_from_stats(stats)
        #stoch = dist.base_dist.mode() # DEBUG
        stoch = dist.sample() if sample else dist.mode() # TODO: This case will fail, since pytorch dists don't have mode
        prior = StateData(stoch=stoch, deter=deters[-1], **stats.to_dict())
        return prior

from typing import Tuple, List
import torch


class StateData:
    """"
    Encapsulates data associated with the internal state of the RSSM
    """

    def __init__(self,
                 stoch=None,
                 deter=None,
                 logit=None,
                 std=None,
                 mean=None,
                 **kwargs):
        self.stoch = stoch
        self.deter = deter
        self.logit = logit
        self.std = std
        self.mean = mean
        [setattr(self, k, v) for k, v in kwargs.items()]

    def __str__(self):
        s = "StateData instance contains:\n"
        for i, (k, v) in enumerate(vars(self).items()):
            if v != None:
                s += f"\t{k} - shape: {v.shape}, type: {v.dtype}" + "\n"
        return s

    def to_dict(self):
        """
            Returns a dict of variables contained in self, except those that are None
        """
        return {k: v for k, v in vars(self).items() if not v is None}

    def requires_grad(self, flag : bool):
        """
            Sets the requires_grad flag of all entries of self
        """
        for v in self.to_dict().values():
            v.requires_grad = flag

    def detach(self, inplace=True):
        """
            Detaches all entries of self
        """
        if inplace:
            for k, v in self.to_dict().items():
                setattr(self, k, v.detach())
        else:
            new = StateData()
            for k, v in self.to_dict().items():
                setattr(new, k, v.detach())
            return new

    @classmethod
    def get_initial_state(cls, discrete, batch_size, stoch_size,
                          deterministic_state_size, device="cpu", seq_len=None):
        stoch, deter, logit, std, mean = None, None, None, None, None
        seq_len = [seq_len] if not seq_len is None else []

        if discrete:
            logit = torch.zeros(seq_len + [batch_size, stoch_size, discrete],
                                dtype=torch.float, device=device)
            stoch = torch.zeros(seq_len + [batch_size, stoch_size, discrete],
                                dtype=torch.float, device=device)
        else:
            mean = torch.zeros(seq_len + (batch_size, stoch_size), dtype=torch.float, device=device)
            std = torch.zeros(seq_len + (batch_size, stoch_size), dtype=torch.float, device=device)
            stoch = torch.zeros(seq_len + (batch_size, stoch_size), dtype=torch.float, device=device)

        # NOTE: Slight deviation from original paper, as they use GRUCell initial
        #       state. However, looking at tensorflow docs. and source code it
        #       seems to just initialize to zeros anyway. See:
        #       https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell#get_initial_state
        #       https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/recurrent.py#L1994-L1995
        deter = torch.zeros(seq_len + [batch_size, deterministic_state_size],
                            dtype=torch.float, device=device)
        return cls(stoch, deter, logit, std, mean)

    @classmethod
    def stack(cls, stats: List['StateData'], dim=0, stack_func=torch.stack) -> 'StateData':
        """
        Assumes all instances of input list have same attributes set to None or
        some tensor of the same length

        Stack func should be either stack or concat depending on wether one of the states already has been stacked 
        """
        stack_keys = [k for k, v in vars(stats[0]).items() if not v is None]
        stacked_data = {
            key: stack_func([getattr(sd, key) for sd in stats], dim=dim)
            for key in stack_keys
        }
        return StateData(**stacked_data)

    @classmethod
    def unstack(cls, stats: 'StateData') -> List['StateData']:
        """
            inverse of StateData.stack(stats, dim=0), does not work if it was stacked along any other dimension
        """
        stack_keys = [k for k, v in vars(stats).items() if not v is None]
        stack_len = getattr(stats, stack_keys[0]).shape[0]
        unstacked_data = [{key: getattr(stats, key)[i]
                           for key in stack_keys} for i in range(stack_len)]
        unstacked_states = [StateData(**data) for data in unstacked_data]
        return unstacked_states
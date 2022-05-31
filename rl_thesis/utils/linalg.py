import torch
from typing import List


def global_norm(tensors : List[torch.tensor]) -> float:
  """
    Given a tuple or list of tensors t_list, this operation returns the global norm of the elements in all tensors in t_list. The global norm is computed as:

    global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

    Any entries in t_list that are of type None are ignored
    https://www.tensorflow.org/api_docs/python/tf/linalg/global_norm
  """
  if isinstance(tensors, torch.Tensor) and not len(tensors.shape) == 2:
    raise TypeError(f"Global norm expects to receive a sequence of tensors, got: {tensors}")
  l2norms_squared = torch.tensor([torch.norm(x, p=2)**2 for x in tensors if not x is None])
  return torch.sqrt(l2norms_squared.sum().type(torch.float32))

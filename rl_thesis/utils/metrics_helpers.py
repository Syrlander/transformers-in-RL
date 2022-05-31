import torch
from .hooks import HookFunction, BackwardHookFunction
from .linalg import global_norm

def append_value_to_metrics(metrics, key, value):
  if key in metrics:
      metrics[key].append(value)
  else:
      metrics[key] = [value]

def append_norm_of_grad_to_metrics(metrics, layer_name, grad):
    append_value_to_metrics(metrics, f"{layer_name}_gradient_norm", torch.norm(grad, p=2).detach().cpu())

def append_global_norm_of_grad_to_metrics(metrics, layer_name, grad):
    append_value_to_metrics(metrics, f"{layer_name}_gradient_norm", global_norm(grad).detach().cpu())

class AppendNormOfGradToMetricsHook(HookFunction):
    def __init__(self, model_name, metrics):
        self.model_name = model_name
        self.metrics = metrics

    def __call__(self, layer_name, grad):
        # print(self.model_name, layer_name)
        # print(grad)
        # print(grad.shape)
        append_norm_of_grad_to_metrics(self.metrics, f"{self.model_name}_{layer_name}", grad)

class AppendGlobalNormOfGradToMetricsBackwardHook(BackwardHookFunction):
    def __init__(self, model_name, metrics):
        self.model_name = model_name
        self.metrics = metrics

    def __call__(self, layer_name, _, grad_input, grad_output):
        # print(self.model_name, layer_name)
        # print(grad_input)
        append_global_norm_of_grad_to_metrics(self.metrics, f"{self.model_name}_{layer_name}", grad_input)

    def add_model_identifier(self, sub_model):
        """
            Returns a new backward hook function that does the same, where model name is f"{self.model_name}_{sub_model}"
        """
        new = AppendGlobalNormOfGradToMetricsBackwardHook(f"{self.model_name}_{sub_model}", self.metrics)
        return new
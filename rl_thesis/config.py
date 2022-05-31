from copy import deepcopy
from pathlib import Path

class BaseConfig:
  def to_json_serializable_dict(self):
        return self.get_vars(self)

  def get_vars(self, obj):
      variables = deepcopy(vars(obj))
      for k, v in variables.items():
          if hasattr(v, "__dict__"):
              variables[k] = self.get_vars(v)
          if isinstance(v, Path):
              variables[k] = str(v)
      return variables
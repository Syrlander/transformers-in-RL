import json


def overwrite_default_values(json_path, config_obj):
  """
    Sets the values of varables in config_obj to match those in the given json file (works IN-PLACE!)
    Raises ValueError if json file contains variables not in config_obj
  """
  with open(json_path, "r") as f:    
    parameters = json.load(f) 

  overwrite_default_values_with_dict(parameters, config_obj)


def overwrite_default_values_with_dict(new_values, config_obj):
  for key, val in new_values.items():
    if key not in vars(config_obj):
      raise ValueError(f"Key {key} was present in config file, but was not found in config class")
    if type(val) is dict and not type(getattr(config_obj, key)) is dict:
      # dig down one level in the nested config structure
      overwrite_default_values_with_dict(val, getattr(config_obj, key))
    else:
      setattr(config_obj, key, val)

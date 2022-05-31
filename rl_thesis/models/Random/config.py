from rl_thesis.models import BaseModelConfig


class Config(BaseModelConfig):
    def __init__(self):
        super().__init__()
        [
            setattr(self, k, v) 
            for k, v in vars(Config).items() 
               if not k.startswith("_")
        ]

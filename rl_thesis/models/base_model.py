class BaseModel:
    def __init__(self, conf, env, policy, policy_config):
        self.conf = conf
        self.model = None

    @classmethod
    def load(self, path, env=None, device="cpu"):
        pass

    def train(self, eval_env):
        pass

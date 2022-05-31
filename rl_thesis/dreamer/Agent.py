import torch
from torch import nn
from rl_thesis.dreamer import WorldModel, ActorCritic, dists


class Agent(nn.Module):
    def __init__(self, config, obs_space, act_space, batch_size, seq_len, device="cpu", hook_metrics=None) -> None:
        super().__init__()
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.wm = WorldModel(config.world_model, obs_space, batch_size, seq_len, hook_metrics)
        
        self._task_behavior = ActorCritic(config.actor_critic, self.act_space, hook_metrics=hook_metrics)
        self.device = device
        self._predict_state = None

        if config.expl_behavior == "greedy":
            self._expl_behavior = self._task_behavior
        else:
            raise NotImplementedError("Can only use greedy exploration behavior.")

    def _action_noise(self, action, amount, act_space):
        if amount == 0:
            return action
        if hasattr(act_space, "n"):
            probs = amount / action.shape[-1] + (1 - amount) * action
            return dists.OneHotDist(probs=probs).sample()
        else:
            return torch.clamp(torch.distributions.Normal(action, amount).sample(), -1, 1)

    def policy(self, obs, state=None, mode="train"):
        obs = {k: torch.Tensor(v).to(self.device) for k, v in obs.items()}
        
        if state is None:
            latent = self.wm.ssm.initialize_state(batch_size=len(obs["reward"]))
            if self.config.world_model.recurrence_model.upper() == "RSSM":
                actions = torch.zeros((len(obs["reward"]), ) + self.act_space.shape, device=self.device)
            elif self.config.world_model.recurrence_model.upper() == "TSSM":
                actions = torch.zeros((1,) + (len(obs["reward"]), ) + self.act_space.shape, device=self.device)
            else:
                self.raise_recurrence_model_error()
            state = latent, actions
        
        latent, actions = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        sample = (mode == "train") or not self.config.eval_state_mean
        # TODO: keep track of obs (or embedding) and actions from current episode then call observe below, I think?
        latent, _ = self.wm.ssm.observation_step(embed, actions, latent, obs["is_first"].type(torch.int), sample)
        # NOTE: GOT TO THE BELOW LINE IN DEBUGGING COMPARISON WITH DREAMER
        if self.config.world_model.recurrence_model.upper() == "RSSM":
            feat = self.wm.ssm.get_feat(latent)
        elif self.config.world_model.recurrence_model.upper() == "TSSM":
            feat = self.wm.ssm.get_feat(latent)[-1]
        else:
            self.raise_recurrence_model_error()
        
        
        if mode == "eval":
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
            noise = self.config.eval_noise    
        elif mode == "explore":
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        elif mode == "train":
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
            noise = self.config.expl_noise
        action = self._action_noise(action, noise, self.act_space)
        outputs = {"action": action.detach().cpu().numpy().reshape((-1,) + self.act_space.shape)}
        if self.config.world_model.recurrence_model.upper() == "RSSM":
            state = (latent, action)
        elif self.config.world_model.recurrence_model.upper() == "TSSM":
            actions = torch.concat((actions,action.unsqueeze(0)), 0)
            state = (latent, actions)
        else:
            self.raise_recurrence_model_error()    
        return outputs, state

    def raise_recurrence_model_error(self):
        raise ValueError(f"Recurrence model {self.config.world_model.recurrence_model} not recognized.")
    def predict(self, obs, deterministic=True):
        output, self._predict_state = self.policy(obs, state=self._predict_state, mode="eval")
        return output, self._predict_state

    def learn(self, data, state=None):
        """
        data:
            ram: [16, 50, 4], dtype: float32
            reward: [16, 50], dtype: float32
            is_first: [16, 50], dtype: bool
            is_last: [16, 50], dtype: bool
            is_terminal: [16, 50], dtype: bool
            action: [16, 50, 2], dtype: float32
        """
        metrics = {}
        state, outputs, mets = self.wm.learn(data, state)
        
        metrics.update(mets)
        start = outputs["post"]
        reward_fn = lambda seq: self.wm.heads["reward"](seq["feat"]).base_dist.loc
        metrics.update(
            self._task_behavior.learn(self.wm, start, data["is_terminal"], reward_fn)
        )
        if self.config.expl_behavior != "greedy":
            mets = self._expl_behavior.learn(self.wm, start, outputs, data)[-1]
            metrics.update(
                {f"expl_{key}" : value for key, value, in mets.items()}
            )

        return state, metrics

    def report(self, data):
        report = {}
        data = self.wm.preprocess(data)
        for key in self.wm.heads["decoder"].cnn_keys:
            name = key.replace('/', '_')
            report[f"openl_{name}"] = self.wm.video_pred(data, key)
        return report

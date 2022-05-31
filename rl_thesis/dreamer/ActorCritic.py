import torch
from torch import nn, optim
from rl_thesis.dreamer import MLPWithDist, StreamNorm, WorldModel
from rl_thesis.dreamer.state_data import StateData
from rl_thesis.dreamer.torch_utils import FreezeParameters
from rl_thesis.utils import linalg, metrics_helpers

class ActorCritic(nn.Module):

    def __init__(self, config, act_space, hook_metrics) -> None:
        super().__init__()
        self.config = config
        self.act_space = act_space
        self.use_autocast = config.use_autocast
        self.actor_scaler = torch.cuda.amp.GradScaler(enabled=self.use_autocast)
        self.critic_scaler = torch.cuda.amp.GradScaler(enabled=self.use_autocast)
        discrete = hasattr(act_space, "n")  # number actions

        if self.config.actor.dist == "auto":
            self.config.actor.dist = "onehot" if discrete else "trunc_normal"
        if self.config.actor_grad == "auto":
            self.config.actor_grad = "reinforce" if discrete else "dynamics"

        # actor = action model
        # TODO: find a way to log actor gradients in a seperate metrics, but still return it??
        # actor_hook = metrics_helpers.AppendNormOfGradToMetricsHook("actor", metrics)
        if not hook_metrics is None: 
            actor_hook_fn = metrics_helpers.AppendGlobalNormOfGradToMetricsBackwardHook("Actor", hook_metrics)
            critic_hook_fn = metrics_helpers.AppendGlobalNormOfGradToMetricsBackwardHook("Critic", hook_metrics)
        else:
            actor_hook_fn = None
            critic_hook_fn = None

        self.actor = MLPWithDist(output_dim=(act_space.shape[0],), **self.config.actor, hook_fn=actor_hook_fn)
        # critic = value model
        self.critic = MLPWithDist(**self.config.critic, hook_fn=critic_hook_fn)

        if self.config.slow_target:
            self._target_critic = MLPWithDist(**self.config.critic)
            self._updates = 0  # in tensorflow code, this is registered as Variable, not sure why, as it does not seem to require gradients or anything
        else:
            self._target_critic = self.critic

        self.actor_opt = optim.AdamW(self.actor.parameters(),
                                     **self.config.actor_opt.kw_args)
        self.critic_opt = optim.AdamW(self.critic.parameters(),
                                      **self.config.critic_opt.kw_args)
        self.reward_normalizer = StreamNorm(**self.config.reward_normalizer)

    def learn(self, world_model: WorldModel, start: StateData, is_terminal,
              reward_fn):
        """
        start:
            * deter: [16, 50, 600]
            * logit: [16, 50, 32, 32]
            * stoch: [16, 50, 32, 32]
        is_terminal: [16, 50] (dtype: bool)
        reward_fn = lambda seq: self.wm.heads["reward"](seq["feat"]).mean
            * from WorldModel.learn
        """
        metrics = {}
        horizon = self.config.imag_horizon

        start_detached = StateData(**{k: v.detach() for k, v in start.to_dict().items()})
        with FreezeParameters([world_model]): # Includes RSSM, encoder and all other heads
            with torch.cuda.amp.autocast(enabled=self.use_autocast):
                # seq:
                #   * stoch: requires_grad = True
                #   * deter: requires_grad = True
                #   * logit: requires_grad = True
                #   * feat: requires_grad = True
                #   * action: requires_grad = True
                #   * discount: requires_grad = True
                #   * weight: requires_grad = False
                #
                # seq (dict):
                #   * stoch: shape:     [16, 800, 32, 32], dtype: float32
                #   * deter: shape:     [16, 800, 600], dtype: float32
                #   * logit: shape:     [16, 800, 32, 32], dtype: float32
                #   * feat: shape:      [16, 800, 1624], dtype: float32
                #   * action: shape:    [16, 800, 2], dtype: float32
                #   * discount: shape:  [16, 800], dtype: float32
                #   * weight: shape:    [16, 800], dtype: float32
                #   * reward: shape:    [16, 800], dtype: float32
                seq = world_model.imagine(self.actor, start_detached, is_terminal, horizon)
                
                # print(f"OURS - seq['deter']: {seq['deter']} (shape: {seq['deter'].shape})")

                # this hook is never called, don't know if that is expected or not
                seq["feat"].register_hook(lambda x: metrics_helpers.append_value_to_metrics(metrics, "feat_gradient", torch.norm(x, p=2).detach().cpu()))
        with FreezeParameters([world_model, self.critic, self._target_critic]):
            with torch.cuda.amp.autocast(enabled=self.use_autocast):
                reward = reward_fn(seq) # requires_grad = True
                seq["reward"], reward_metrics = self.reward_normalizer(reward.detach()) # seq['reward']: requires_grad = True (matches the above reward variable - no normalization applied it seems. TODO: Maybe delete the reward_normalizer???)
                reward_metrics = {f"reward_{k}": v.detach().cpu() for k, v in reward_metrics.items()}

                # target: tensor - shape: [15, 800], dtype: float32
                target, target_metrics = self.target(seq) # target (lambda_returns): requires_grad = True

                actor_loss, actor_loss_metrics = self.actor_loss(seq, target) # actor_loss: requires_grad = True
        with torch.cuda.amp.autocast(self.use_autocast):
            critic_loss, critic_loss_metrics = self.critic_loss(seq, target) # critic_loss: requires_grad = True
        
        # print(f"OURS - seq: {seq}")
        # print(f"OURS - target: {target}")
        # print(f"OURS - reward: {reward}")

        # print(f"OURS - actor_loss: {actor_loss}")
        # print(f"OURS - critic_loss: {critic_loss}")

        self.actor_opt.zero_grad()
        self.actor_scaler.scale(actor_loss).backward()
        # actor_loss.backward() # DEBUG
        
        # def print_grads(l):
        #     if isinstance(l, nn.Linear):
        #         print(l)
        #         print(f"weight grad: {l.weight.grad}")
        #         print(f"bias grad: {l.bias.grad}")

        # print("OURS - ACTOR GRADS")
        # self.actor.apply(print_grads)

        self.critic_opt.zero_grad()
        # critic_loss.backward() # DEBUG
        self.critic_scaler.scale(critic_loss).backward()
        

        # gradients should be unscaled before clipping https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-scaled-gradients 
        self.critic_scaler.unscale_(self.critic_opt)
        self.actor_scaler.unscale_(self.actor_opt)
        actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                       self.config.actor_opt.clip)
        critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                       self.config.critic_opt.clip)

        metrics["actor_gradient_norm_clip_ret"] = actor_norm.detach().cpu().item()
        metrics["critic_gradient_norm_clip_ret"] = critic_norm.detach().cpu().item()

        self.actor_scaler.step(self.actor_opt)
        self.critic_scaler.step(self.critic_opt)
        self.actor_scaler.update()
        self.critic_scaler.update()

        metrics["actor_gradients"] = linalg.global_norm(self.actor.parameters()).detach().cpu()
        metrics["critic_gradients"] = linalg.global_norm(self.critic.parameters()).detach().cpu() 

        metrics.update(
            **reward_metrics,
            **target_metrics,
            **actor_loss_metrics,
            **critic_loss_metrics,
        )
        self.update_slow_target()
        return metrics

    def actor_loss(self, seq, target):
        metrics = {}

        policy = self.actor(seq["feat"][:-2].detach())
        # print(policy)
        if self.config.actor_grad == "dynamics":
            objective = target[1:]
        elif self.config.actor_grad == "reinforce":
            baseline = self._target_critic(seq["feat"][:-2]).mean
            advantage = (target[1:] - baseline).detach()
            # NOTE: I removed a detach here because it seems that there are no detaches in reference implementations
            action = seq["action"][1:-1]
            #print(action)
            lp = policy.log_prob(action)
            objective = lp * advantage
            # print(objective)
            # print(objective.shape)
        elif self.config.actor_grad == "both":
            baseline = self._target_critic(seq['feat'][:-2]).mean
            advantage = (target[1:] - baseline).detach()
            objective = policy.log_prob(seq["action"][1:-1]) * advantage
            
            # NOTE: slight divergence from original implementation as they can specify functions here, but they always use a single scala, so we'll do that here as well
            
            # NOTE: The whole thing with mix and objective is Conservative Policy Iteration
            # of Approximately Optimal Approximate Reinforcement Learning, by Kakade & Langford (see page 6)
            # 
            # Also see: Trust Region Policy Optimization (TRPO), by Schulman et al. (page 2, eq. 5)
            # for a bit on this.
            mix = self.config.actor_grad_mix
            objective = mix * target[1:] + (1 - mix) * objective
            metrics["actor_grad_mix"] = mix
        else:
            raise NotImplementedError(f"Got invalid actor grad mode: '{self.config.actor_grad}'")
        metrics["actor_objective"] = objective.mean().detach().cpu()
        metrics["actor_advantage"] = advantage.mean().detach().cpu()
        metrics["actor_log_prob"] = lp.mean().detach().cpu()
        ent = policy.entropy()
        ent_scale = self.config.actor_ent
        objective += ent_scale * ent
        weight = seq["weight"].detach()
        actor_loss = -(weight[:-2] * objective).mean()
        metrics["actor_entropy"] = ent.mean().detach().cpu()
        metrics["actor_entropy_scale"] = ent_scale
        metrics["actor_loss"] = actor_loss.detach().cpu()
        return actor_loss, metrics

    def critic_loss(self, seq, target):
        dist = self.critic(seq["feat"][:-1].detach()) 
        target = target.detach()
        weight = seq["weight"].detach()
        critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
        metrics = {"critic": dist.mean.mean().detach().cpu()}
        metrics["critic_loss"] = critic_loss.detach().cpu()
        return critic_loss, metrics

    def lambda_return(self, reward, value, pcont, bootstrap, lambda_, axis):
        """
        Lambda discounted return over a batch of data

        pcont contains the discount
        """
        # Setting lambda=1 gives a discounted Monte Carlo return.
        # Setting lambda=0 gives a fixed 1-step return.
        assert reward.ndim == value.ndim, (reward.shape, value.shape)
        if isinstance(pcont, (int, float)):
            pcont = pcont * torch.ones_like(reward)
        dims = list(range(reward.ndim))
        dims = [axis] + dims[1:axis] + [0] + dims[axis + 1:]
        if axis != 0:
            reward = torch.transpose(reward, dims)
            value = torch.transpose(value, dims)
            pcont = torch.transpose(pcont, dims)
        if bootstrap is None:
            bootstrap = torch.zeros_like(value[-1])
        next_values = torch.concat([value[1:], bootstrap[None]], 0)
        inputs = reward + pcont * next_values * (1 - lambda_)

        # Static scan equivalent
        agg, last = [], bootstrap
        for c1, c2 in zip(reversed(inputs), reversed(pcont)):
            last = c1 + c2 * lambda_ * last
            agg.append(last)
        returns = torch.flip(torch.stack(agg), [0])  #list(reversed(agg))

        if axis != 0:
            returns = torch.transpose(returns, dims)
        return returns

    def target(self, seq):
        reward =  seq["reward"].type(torch.float32)
        discount = seq["discount"].type(torch.float32)
        value = self._target_critic(seq["feat"]).mean #NOTE: mean = mode for Normal dist.
        # NOTE: Check that all arguments have requires grad true
        target = self.lambda_return(reward[:-1],
                                      value[:-1],
                                      discount[:-1],
                                      bootstrap=value[-1],
                                      lambda_=self.config.discount_lambda,
                                      axis=0)

        metrics = {"critic_slow": value.mean().detach().cpu(), "critic_target": target.mean().detach().cpu()}
        return target, metrics

    def update_slow_target(self):
        if self.config.slow_target:
            if self._updates % self.config.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.config.slow_target_fraction)
                for s, d in zip(self.critic.parameters(),
                                self._target_critic.parameters()):
                    d = mix * s + (1 - mix) * d
            self._updates += 1

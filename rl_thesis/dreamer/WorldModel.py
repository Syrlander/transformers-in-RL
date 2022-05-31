import torch
from torch import nn
from rl_thesis.dreamer.torch_utils import get_optim
from rl_thesis.dreamer import Encoder, Decoder, MLPWithDist, StateData
from rl_thesis.dreamer import state_space_models as ssm
from rl_thesis.utils import linalg, metrics_helpers

class WorldModel(nn.Module):

    def __init__(self, config, obs_space, batch_size, seq_len, hook_metrics) -> None:
        super(WorldModel, self).__init__()
        shapes = {k: v.shape for k, v in obs_space.items()}
        self.config = config
        self.use_autocast = config.use_autocast
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_autocast)
        if not hook_metrics is None:
            rssm_hook = metrics_helpers.AppendGlobalNormOfGradToMetricsBackwardHook("RSSM", hook_metrics)
            encoder_hook = metrics_helpers.AppendGlobalNormOfGradToMetricsBackwardHook("encoder", hook_metrics)
            decoder_hook = metrics_helpers.AppendGlobalNormOfGradToMetricsBackwardHook("decoder", hook_metrics)
            reward_hook = metrics_helpers.AppendGlobalNormOfGradToMetricsBackwardHook("reward", hook_metrics)
            discount_hook = metrics_helpers.AppendGlobalNormOfGradToMetricsBackwardHook("discount", hook_metrics)
        else:
            rssm_hook, encoder_hook, decoder_hook, reward_hook, discount_hook = (None for _ in range(5))
        ssm_class = getattr(ssm, config.recurrence_model.upper())
        self.ssm = ssm_class(seq_len, batch_size, **config.rssm, hook_fn=rssm_hook)
        self.encoder = Encoder(shapes, **config.encoder, hook_fn=encoder_hook)
        self.heads = nn.ModuleDict({
            "decoder": Decoder(shapes, **config.decoder, hook_fn=decoder_hook),
            "reward": MLPWithDist(**config.reward_head, hook_fn=reward_hook),
        })

        if config.pred_discount:
            self.heads["discount"] = MLPWithDist(**config.discount_head, hook_fn=discount_hook)

        for name in config.grad_heads:
            assert name in self.heads, name

        # Clipping is handled in train. To add weight decay to e.g. Adam
        # use AdamW instead, see: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        self.model_optimizer = get_optim(config.model_opt.name, self.parameters(), **config.model_opt.kw_args)

    def learn(self, data, state=None):
        model_loss, state, outputs, metrics = self.loss(data, state)
        self.model_optimizer.zero_grad()
        self.grad_scaler.scale(model_loss).backward()

        # unscale optimizer's grads before logging and clipping grads
        self.grad_scaler.unscale_(self.model_optimizer)

        metrics["model_gradients"] = linalg.global_norm(self.parameters()).detach().cpu()
        model_norm = torch.nn.utils.clip_grad_norm_(self.parameters(),
                                       self.config.model_opt.clip)
        metrics["model_gradient_norm_clip_ret"] = model_norm.detach().cpu().item()
        metrics["model_gradients_post_clip"] = linalg.global_norm(self.parameters()).detach().cpu()

        self.grad_scaler.step(self.model_optimizer)
        self.grad_scaler.update()
        return state, outputs, metrics

    def loss(self, data, state=None):
        data = self.preprocess(data)
        metrics = {}
        with torch.cuda.amp.autocast(enabled=self.use_autocast):
            embed = self.encoder(data)

            prior, post = self.ssm.observe(embed, data["action"],
                                            data["is_first"], state)
            kl_loss, kl_value = self.ssm.kl_loss(post, prior, **self.config.kl)
            assert len(kl_loss.shape) == 0

            likes = {}
            losses = {"kl": kl_loss}
            feat = self.ssm.get_feat(post)
            for name, head in self.heads.items():
                grad_head = (name in self.config.grad_heads)
                inp = feat if grad_head else feat.detach()
                out = head(inp)
                dists = out if isinstance(out, dict) else {name: out}
                for key, dist in dists.items():
                    curr_data = data[key]
                    
                    
                    # if key == "discount":
                    #     curr_data = (curr_data == 0).type(torch.float32)

                    like = dist.log_prob(curr_data)
                    likes[key] = like
                    losses[key] = -like.mean()
            model_loss = sum(
                self.config.loss_scales.get(key, 1.0) * value
                for key, value in losses.items())
            outs = {
                # "embed": embed,
                # "feat": feat,
                "post": post,
                # "prior": prior,
                # "likes": likes,
                # "kl": kl_value,
            }
        metrics.update({f"{name}_loss": value.detach().cpu() for name, value in losses.items()})
        metrics["model_kl"] = kl_value.mean().detach().cpu()
        metrics["prior_entropy"] = self.ssm.distribution_from_stats(
            prior).entropy().mean().detach().cpu()
        metrics["post_entropy"] = self.ssm.distribution_from_stats(
            post).entropy().mean().detach().cpu()
        last_state = StateData(
            **{key: value[:, -1].detach()
               for key, value in post.to_dict().items()})
        return model_loss, last_state, outs, metrics

    def imagine(self, policy, start: StateData, is_terminal, horizon, metrics=None):
        """
            If metrics is given it will be passed to rssm.imagine_step
        """
        flatten = lambda x: x.view((-1, ) + x.shape[2:])
        start = {k: flatten(v) for k, v in start.to_dict().items()}
        start = StateData(**start) # all requires_grad = False
        # set new property of start
        start.feat = self.ssm.get_feat(start) # feat: requires_grad = False
        start.action = torch.zeros_like(policy(start.feat).mode()) # action: requires_grad = False
        seq = {k: [v] for k, v in start.to_dict().items()}
        for _ in range(horizon):
            # CHANGE 
            action = policy(seq["feat"][-1].detach()).mode()
            # action = policy(seq["feat"][-1].detach()).sample() # requires_grad = True
            #action = policy(seq["feat"][-1].detach()).mode() # DEBUG
            #print(f"[imagine] OUR - action: {action} (shape: {action.shape})")
            #print(f"OURS - prev_state: {prev_state.to_dict()}")
            if self.config.recurrence_model.upper() == "RSSM":
                prev_state = StateData(**{k: v[-1] for k, v in seq.items()}) # all requires_grad = False
                state = self.ssm.imagine_step(prev_state, action)
            else:
                prev_states = StateData(**{k: torch.stack(v, 0) for k, v in seq.items()})
                state = self.ssm.imagine_step(prev_states, torch.concat([prev_states.action[1:], action.unsqueeze(0)], dim=0))
            # NOTE: Problem seems to be related to feat, if you detach we no longer have mem issues. Detaching just the input state seems to also fix issues, don't know if that breaks something else
            # feat = self.rssm.get_feat(state.detach(inplace=False))
            feat = self.ssm.get_feat(state)
            # print(f"world model.imagine.feat.requires_grad {feat.requires_grad}")
            state.feat = feat
            state.action = action
            for key, value in state.to_dict().items():
                seq[key].append(value)
        seq = {k: torch.stack(v, 0) for k, v in seq.items()}

        if "discount" in self.heads:
            disc = self.heads["discount"](seq["feat"]).mean # requires_grad = False
            if is_terminal is not None:
                # Override discount prediction for the first step with the true
                # discount factor from the replay buffer.
                #true_first = (~flatten(is_terminal)).type(torch.float32)
                true_first = 1.0 - flatten(is_terminal).type(torch.float32)
                true_first *= self.config.discount
                disc = torch.concat((true_first[None], disc[1:]), 0)
        else:
            disc = self.config.discount * torch.ones(seq["feat"].shape[:-1])
        seq["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.

        # TODO: Weights might need to be detached here - if the discount has requires_grad = True
        seq["weight"] = torch.cumprod(
            torch.concat((torch.ones_like(disc[:1]), disc[:-1]), 0), 0).detach()
        return seq # only action has requires_grad = True, all other have False

    def preprocess(self, obs):
        dtype = torch.float32
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_"):
                continue
            #value = torch.tensor(value)
            if value.dtype == torch.int32:
                value = value.type(dtype)
            if value.dtype == torch.uint8:
                value = value.type(dtype) / 255.0 - 0.5
            obs[key] = value
        obs['reward'] = {
            'identity': lambda x: x,
            'sign': torch.sign,
            'tanh': torch.tanh,
        }[self.config.clip_rewards](obs['reward'])
        #obs['discount'] = (~(obs['is_terminal'].type(torch.int16))).type(dtype)
        obs['discount'] = 1.0 - obs["is_terminal"].type(dtype)
        obs['discount'] *= self.config.discount
        return obs

    def video_pred(self, data, key):
        decoder = self.heads['decoder']
        truth = data[key][:6] + 0.5
        embed = self.encoder(data)
        _, states = self.ssm.observe(embed[:6, :5], data['action'][:6, :5],
                                      data['is_first'][:6, :5])
        recon = decoder(self.ssm.get_feat(states))[key].mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.ssm.imagine(data['action'][:6, 5:], init)
        openl = decoder(self.ssm.get_feat(prior))[key].mode()
        model = torch.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
        error = (model - truth + 1) / 2
        video = torch.concat([truth, model, error], 2)

        # NOTE: We may run into some issues here, since we have worked with the color channel being before the height and width
        B, T, H, W, C = video.shape
        return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

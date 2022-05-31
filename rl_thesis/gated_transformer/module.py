import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
from collections import OrderedDict
from .mem_transformer import MemTransformer
from .model import ImgEncoder
from tqdm import tqdm


class StateRepresentation(nn.Module):
    def __init__(self, 
        state_rep, 
        action_dim, 
        observation_dim, 
        n_head, 
        n_layer, 
        n_latent_var, 
        encoder="ResNet", 
        mlp_encoder_layers=4, 
        mlp_encoder_dim=400, 
        mem_len=10, 
        dropout=0.0,
        device="cpu"
    ):
        super(StateRepresentation, self).__init__()

        self.state_rep = state_rep
        self.action_dim = action_dim

        self.encoder_type = encoder
        self.mem_len = mem_len

        self.device = device

        inp_dim = n_latent_var + action_dim + 1 # current state, previous action and reward
        out_dim = n_latent_var

        if encoder == "ResNet":
            self.encoder = ImgEncoder(G=1, img_enc_dim=n_latent_var)
        elif encoder == "MLP":
            # RAM encoder
            encoder_layers = OrderedDict([
                ("dense1", nn.Linear(observation_dim, mlp_encoder_dim)),
                ("act1", nn.ELU()),
            ])
            for i in range(2, mlp_encoder_layers):
                encoder_layers[f"dense{i}"] = nn.Linear(mlp_encoder_dim, mlp_encoder_dim)
                encoder_layers[f"act{i}"] = nn.ELU()

            encoder_layers[f"dense{mlp_encoder_layers}"] = nn.Linear(mlp_encoder_dim, n_latent_var)
            encoder_layers[f"act{mlp_encoder_layers}"] = nn.ELU()

            self.encoder = nn.Sequential(encoder_layers)
        else:
            raise ValueError(f"Got invalid encoder argument: {encoder}.\n\nAvailable encoders are:\n\t'ResNet'\n\t'MLP'\n\t\t* If using MLP encoder remember to also set the --mlp_encoder_dim and --mlp_encoder_layers")

        if state_rep == 'lstm':
            self.layer = nn.LSTMCell(inp_dim, out_dim)
            self.h0 = nn.Parameter(torch.rand(n_latent_var))
            self.c0 = nn.Parameter(torch.rand(n_latent_var))
        elif state_rep == 'transformer':
            self.layer = MemTransformer(
                    inp_dim, n_layer=n_layer, n_head=n_head,
                    d_model=n_latent_var,
                    d_head=n_latent_var // n_head, d_inner=n_latent_var,
                    dropout=dropout, dropatt=dropout, pre_lnorm=False,
                    gating=False, tgt_len=1, ext_len=0, mem_len=0, attn_type=2)
        elif state_rep == 'trxl':
            self.layer = MemTransformer(
                    inp_dim, n_layer=n_layer, n_head=n_head,
                    d_model=n_latent_var,
                    d_head=n_latent_var // n_head, d_inner=n_latent_var,
                    dropout=dropout, dropatt=dropout, pre_lnorm=False,
                    gating=False, tgt_len=1, ext_len=0, mem_len=mem_len, attn_type=0)
        elif state_rep == 'gtrxl':
            self.layer = MemTransformer(
                    inp_dim, n_layer=n_layer, n_head=n_head,
                    d_model=n_latent_var,
                    d_head=n_latent_var // n_head, # Matches the head dim. of 64 (of numpad test in appendix c. of Parisotto et al.) if n_latent_var = 256 and n_head = 8
                    d_inner=n_latent_var,
                    dropout=dropout, dropatt=dropout, pre_lnorm=True,
                    gating=True, tgt_len=1, ext_len=0, mem_len=mem_len, attn_type=0)

        self.init_action = nn.Parameter(torch.rand(action_dim))
        self.init_reward = nn.Parameter(torch.rand(1))

    def forward(self, t, obs, start_of_trajectory, _prev_action=None, _prev_reward=None):
        """
        Args:
            obs: observation (image shape = (64, 64, 3) or ram shape = (env. obs_dim))
        """
        if self.encoder_type == "ResNet":
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0).permute(0,3,1,2)
        elif self.encoder_type == "MLP":
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Got invalid encoder type: {self.encoder_type}")

        state = self.encoder(obs).squeeze() # shape = n_latent_var

        if self.state_rep == 'none':
            return state

        if start_of_trajectory:
            prev_action = self.init_action
            prev_reward = self.init_reward
            if self.state_rep == 'lstm':
                self.h = self.h0.unsqueeze(0)
                self.c = self.c0.unsqueeze(0)
            if self.state_rep == 'transformer':
                self.inputs = []
                self.mems = None
            if self.state_rep in ['trxl', 'gtrxl']:
                self.inputs = []
                self.mems = tuple()
        else:
            prev_action = torch.zeros(self.action_dim, device=self.device)
            prev_action[_prev_action] = 1
            prev_reward = torch.tensor([_prev_reward], dtype=torch.float, device=self.device)

        # [1, inp_dim]
        inp = torch.cat([state, prev_action, prev_reward], dim=0)
        inp = inp.unsqueeze(0)

        if self.state_rep == 'lstm':
            h, c = self.layer(inp, (self.h, self.c))
            self.h = h
            self.c = c
            return h[0]
        elif self.state_rep == 'transformer':
            self.inputs.append(inp)
            _inputs = torch.stack(self.inputs, dim=0)
            pred, _mems = self.layer(_inputs)
            return pred[0][0]
        elif self.state_rep in ['trxl', 'gtrxl']:
            self.inputs.append(inp)
            _inputs = torch.stack(self.inputs, dim=0)
            pred, _mems = self.layer(_inputs, *self.mems)
            if t >= (1 + self.mem_len):
                self.mems = _mems
            return pred[0][0]

    def batch_forward(self, ts, observations, actions, rewards, unroll_length):
        if self.state_rep == 'none':
            observations = np.array(observations)
            observations = torch.from_numpy(observations).float().to(self.device).permute(0,3,1,2)
            rep_states = self.encoder(observations).squeeze()
        else:
            rep_states = []
            for i in range(len(ts)):
                t = ts[i]
                observation = observations[i]

                start_of_trajectory = i % unroll_length == 0
                if start_of_trajectory:
                    prev_action = None
                    prev_reward = None
                else:
                    prev_action = actions[i-1]
                    prev_reward = rewards[i-1]

                if prev_action == None and prev_reward == None:
                    rep_states.append(self.forward(t, observation, start_of_trajectory))
                else:
                    rep_states.append(self.forward(t, observation, start_of_trajectory, _prev_action=prev_action, _prev_reward=prev_reward))

            rep_states = torch.stack(rep_states, dim=0)

        return rep_states


class ActorCritic(nn.Module):
    def __init__(self, 
        model, 
        state_rep, 
        action_dim, 
        observation_dim,
        n_head=8,
        n_layer=4,
        n_latent_var=32, 
        encoder="ResNet",
        mlp_encoder_layers=None,
        mlp_encoder_dim=None,
        mem_len=10, 
        device="cpu"
    ):
        super(ActorCritic, self).__init__()

        self.model = model
        self.device = device
        self.state_rep = state_rep
        inp_dim = n_latent_var

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(inp_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(inp_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
            )

        # shared state representation module
        self.shared_layer = StateRepresentation(state_rep, action_dim, observation_dim, n_head, n_layer, n_latent_var, encoder=encoder, mlp_encoder_layers=mlp_encoder_layers, mlp_encoder_dim=mlp_encoder_dim, mem_len=mem_len, device=device)

    def forward(self):
        raise NotImplementedError

    def act(self, t, obs, memory):
        if t == 0:
            rep_state = self.shared_layer(t, obs, True)
        else:
            rep_state = self.shared_layer(t, obs, False, memory.prev_action, memory.prev_reward)

        # TODO: Extend to continuous action spaces
        action_probs = self.action_layer(rep_state)
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item()

    def evaluate(self, ts, observations, actions, rewards, unroll_length):
        rep_states = self.shared_layer.batch_forward(ts, observations, actions, rewards, unroll_length)

        action_probs = self.action_layer(rep_states.detach())
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(actions)
        if self.model == 'ppo':
            dist_entropy = dist.entropy()
        elif self.model == 'vmpo':
            dist_probs = dist.probs

        state_value = self.value_layer(rep_states)

        if self.model == 'ppo':
            return action_logprobs, torch.squeeze(state_value), dist_entropy
        elif self.model == 'vmpo':
            return action_logprobs, torch.squeeze(state_value), dist_probs


class VMPO(nn.Module):
    def __init__(self, 
        state_rep, 
        action_dim,
        observation_dim,
        n_head=8,
        n_layer=4,
        encoder="ResNet",
        mlp_encoder_layers=None,
        mlp_encoder_dim=None,
        n_latent_var=32,
        mem_len=10,
        init_alpha=5.0,
        init_eta=1.0,
        eps_eta=0.1,
        lr=0.001, 
        betas=0.999, 
        gamma=0.99,
        eps_alpha=0.01,
        K_epochs=4, 
        device="cpu"
    ):
        super(VMPO, self).__init__()

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs

        self.eta = torch.autograd.Variable(torch.tensor(init_eta), requires_grad=True)
        self.alpha = torch.autograd.Variable(torch.tensor(init_alpha), requires_grad=True)
        self.eps_eta = eps_eta
        self.eps_alpha = eps_alpha
        self.device = device

        self.learning_steps = 0

        self.policy = ActorCritic(
            'vmpo', 
            state_rep, 
            action_dim, 
            observation_dim, 
            n_head=n_head,
            n_layer=n_layer,
            n_latent_var=n_latent_var, 
            encoder=encoder,
            mlp_encoder_layers=mlp_encoder_layers,
            mlp_encoder_dim=mlp_encoder_dim,
            mem_len=mem_len,
            device=device)

        params = [
                {'params': self.policy.parameters()},
                {'params': self.eta},
                {'params': self.alpha}
            ]

        self.optimizer = torch.optim.Adam(params, lr=lr, betas=betas)

        self.policy_old = ActorCritic(
            'vmpo', 
            state_rep, 
            action_dim, 
            observation_dim, 
            n_head=n_head,
            n_layer=n_layer,
            n_latent_var=n_latent_var,
            encoder=encoder,
            mlp_encoder_layers=mlp_encoder_layers,
            mlp_encoder_dim=mlp_encoder_dim, 
            mem_len=mem_len,
            device=device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def get_KL(self, prob1, logprob1, logprob2):
        kl = prob1 * (logprob1 - logprob2)
        return kl.sum(1, keepdim=True)

    def __flatten(self, arr):
        return [item for sublist in arr for item in sublist]

    def update(self, batch):
        metrics = { }

        unroll_length = len(batch["reward"][0])
        # print(f"unroll_length: {unroll_length}")

        # Monte Carlo estimate of state rewards:
        rewards = []
        for trajectory_zip in zip(batch["reward"], batch["done"]):
            discounted_reward = 0
            for reward, is_done in reversed(list(zip(*trajectory_zip))):
                if is_done:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_ts = self.__flatten(batch["timestep"])
        # print(f"old_ts: {old_ts} (len: {len(old_ts)})")
        old_states = self.__flatten(batch["state"])
        # print(f"old_states: {old_states} (len: {len(old_states)})")
        old_actions = torch.tensor(self.__flatten(batch["action"]), dtype=torch.uint8, device=self.device, requires_grad=False)
        # print(f"old_actions: {old_actions} (len: {len(old_actions)})")
        old_rewards = self.__flatten(batch["reward"])
        # print(f"old_rewards: {old_rewards} (len: {len(old_rewards)})")

        # Get old probs and old advantages
        with torch.no_grad():
            _, old_state_values, old_dist_probs = self.policy_old.evaluate(old_ts, old_states, old_actions, old_rewards, unroll_length)
            advantages = rewards - old_state_values.detach()

        # Optimize policy for K epochs:
        for i in tqdm(range(self.K_epochs), desc="Running V-MPO update"):
            # Evaluating sampled actions and values:
            logprobs, state_values, dist_probs = self.policy.evaluate(old_ts, old_states, old_actions, old_rewards, unroll_length)

            # Get samples with top half advantages
            advprobs = torch.stack((advantages, logprobs))
            advprobs = advprobs[:, torch.sort(advprobs[0], descending=True).indices]
            good_advantages = advprobs[0, :len(old_states) // 2]
            good_logprobs = advprobs[1, :len(old_states) // 2]

            # Get losses
            phis = torch.exp(good_advantages / self.eta.detach()) / torch.sum(torch.exp(good_advantages / self.eta.detach()))
            L_pi = -phis * good_logprobs
            L_eta = self.eta * self.eps_eta + self.eta * torch.log(torch.mean(torch.exp(good_advantages / self.eta)))

            KL = self.get_KL(old_dist_probs.detach(), torch.log(old_dist_probs).detach(), torch.log(dist_probs))

            L_alpha = torch.mean(self.alpha * (self.eps_alpha - KL.detach()) + self.alpha.detach() * KL)

            L_v = 0.5 * self.MseLoss(state_values, rewards)

            loss = torch.mean(L_pi + L_eta + L_alpha + L_v)

            metrics[f"vmpo_loss_pi_epoch-{i}"] = L_pi.mean().detach().cpu().numpy()
            metrics[f"vmpo_loss_eta_epoch-{i}"] = L_eta.mean().detach().cpu().numpy()
            metrics[f"vmpo_loss_alpha_epoch-{i}"] = L_alpha.mean().detach().cpu().numpy()
            metrics[f"vmpo_loss_v_epoch-{i}"] = L_v.mean().detach().cpu().numpy()
            metrics[f"vmpo_loss_epoch-{i}"] = loss.detach().cpu().numpy()

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.eta.copy_(torch.clamp(self.eta, min=1e-8))
                self.alpha.copy_(torch.clamp(self.alpha, min=1e-8))

        self.learning_steps += 1

        # Copy new weights into old policy (T_target = 10):
        if self.learning_steps % 10 == 0:
            self.policy_old.load_state_dict(self.policy.state_dict())

        return metrics
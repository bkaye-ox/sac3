import gymnasium as gym
import copy as cp
import itertools as it
import torch
import torch.nn as nn
import common
import numpy as np


class Agent():
    def __init__(self, offpolicy) -> None:
        self.offpolicy = offpolicy

    def add_replay(self, obs, action, reward, next_obs, done):
        raise NotImplementedError if self.offpolicy else Exception(
            'Agent is on-policy')

    def act(self, obs, deterministic=False):
        raise NotImplementedError

    def batch_replay(self):
        raise NotImplementedError if self.offpolicy else Exception(
            'Agent is on-policy')


class ActorCritic(nn.Module):
    def __init__(self, action_space, obs_space, net_size) -> None:
        super().__init__()
        self.action_space = action_space
        self.obs_space = obs_space

        cont = not isinstance(action_space, gym.spaces.Discrete)

        act_out = self.action_space.shape[0] if cont else self.action_space.n
        act_in = act_out if cont else 1

        action_range = (self.action_space.low,
                        self.action_space.high) if cont else (None, None)

        self.actor = Actor(act_out, obs_dim :=
                           obs_space.shape[0], net_size, action_range, cont=cont)
        self.critic_1 = Critic(act_in, obs_dim, net_size)
        self.critic_2 = Critic(act_in, obs_dim, net_size)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(obs, deterministic=deterministic)
            return a.numpy()


class MLP(nn.Module):
    def __init__(self, inp, hidden_shape, out, act=nn.ReLU) -> None:
        super().__init__()
        layer_args = [(inp, hidden_shape[0])]+[(n_in, n_out) for n_in,
                                               n_out in zip(hidden_shape[:-1], hidden_shape[1:])]+[(hidden_shape[-1], out)]
        self.mlp = nn.Sequential(*[mod for n_in, n_out in layer_args for mod in [nn.Linear(n_in, n_out), act()]
                                   ])

    def forward(self, x):
        return self.mlp(x)


class Critic(nn.Module):
    def __init__(self, act_dim, obs_dim, net_size) -> None:
        super().__init__()
        self.critic = MLP(act_dim + obs_dim, net_size, 1)

    def forward(self, obs, act):
        if len(obs.shape) > len(act.shape):
            act = act.unsqueeze(-1)
        x = torch.cat([obs, act], dim=-1)
        q = self.critic(x)
        return q


class Actor(nn.Module):
    def __init__(self, act_dim, obs_dim, net_size, action_range=(None, None), cont=True, dist_trfm=torch.distributions.TanhTransform) -> None:
        super().__init__()
        self.cont = cont

        out_dim = 2*act_dim if cont else act_dim
        self.actor = MLP(obs_dim, net_size, out_dim)

        self.transform = dist_trfm if isinstance(
            dist_trfm, torch.distributions.Transform) else None

        self.action_range = action_range

    def forward(self, obs, deterministic=False):
        if self.cont:
            mu, logvar = self.actor(obs).chunk(2, dim=-1)
            scale = torch.exp(0.5*logvar)
            gauss = torch.distributions.Normal(loc=mu, scale=scale)
            dist = torch.distributions.TransformedDistribution(
                gauss, self.transform) if self.transform is not None else gauss
        else:
            logits = self.actor(obs)
            dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            return mu if self.cont else dist.probs.argmax(dim=-1), None

        a = dist.sample()
        log_prob = dist.log_prob(a)

        a = self.action_range[0] + (1 + a)/2 * (self.action_range[1] -
                                                self.action_range[0]) if self.action_range != (None, None) and self.cont else a

        return a, log_prob


class Sac(Agent):
    def __init__(self, env_info, net_size=[16], batch_size=256) -> None:
        super().__init__(offpolicy=True)

        action_space, obs_space = env_info['action_space'], env_info['obs_space']

        discrete = isinstance(action_space, gym.spaces.Discrete)

        act_dim = action_space.shape[0] if not discrete else action_space.n
        act_out = act_dim if not discrete else 1

        self.ac = ActorCritic(action_space, obs_space,
                              net_size)
        self.ac_target = cp.deepcopy(self.ac)
        for p in self.ac_target.parameters():
            p.requires_grad = False
        self.q_params = it.chain(
            self.ac.critic_1.parameters(), self.ac.critic_2.parameters())
        self.replay = common.ReplayBuffer(
            obs_space.shape[0], act_out, int(1e6))
        self.polyak = 0.995
        self.gamma = 0.99
        self.alpha = 0.05
        self.lr = 1e-4

        self.batch_size = 256

        self.configure_optimizers()

    def configure_optimizers(self):
        self.pi_optim = torch.optim.Adam(
            self.ac.actor.parameters(), lr=self.lr)
        self.q_optim = torch.optim.Adam(self.q_params, lr=self.lr)

    def q_loss(self, batch):
        obs, next_obs, act, rew, done = [v for k, v in batch.items()]
        q1 = self.ac.critic_1(obs, act)
        q2 = self.ac.critic_2(obs, act)

        with torch.no_grad():
            a2, logp_a2 = self.ac.actor(next_obs)

            q1_target, q2_target = [q(next_obs, a2) for q in [
                self.ac_target.critic_1, self.ac_target.critic_2]]
            q_target = torch.min(q1_target, q2_target)

            backup = rew + self.gamma * \
                (1 - done) * (q_target - self.alpha * logp_a2)

        loss = sum([torch.pow(q - backup, 2).mean() for q in [q1, q2]])
        return loss

    def pi_loss(self, batch):
        obs, *rest = [v for k, v in batch.items()]
        a, logp_a = self.ac.actor(obs)

        q1, q2 = [q(obs, a) for q in [
            self.ac.critic_1, self.ac.critic_2]]
        q = torch.min(q1, q2)

        loss = (self.alpha * logp_a - q).mean()
        return loss

    def update(self, batch):
        q_opt = self.q_optim
        q_opt.zero_grad()
        loss_q = self.q_loss(batch)
        loss_q.backward()
        q_opt.step()

        for p in self.q_params:
            p.requires_grad = False

        pi_opt = self.pi_optim
        pi_opt.zero_grad()
        loss_pi = self.pi_loss(batch)
        loss_pi.backward()
        pi_opt.step()

        for p in self.q_params:
            p.requires_grad = True

        for p, p_targ in zip(self.ac.parameters(), self.ac_target.parameters()):
            p_targ.data.mul_(self.polyak)
            p_targ.data.add_((1-self.polyak) * p.data)

    def act(self, obs, deterministic=False):
        a, _ = self.ac.actor(obs, deterministic=deterministic)
        return a.detach().numpy()

    def add_replay(self, obs, action, reward, next_obs, done):
        nan = np.isnan(obs).any() or np.isnan(next_obs).any(
        ) or np.isnan(action).any() or np.isnan(reward).any()
        if not nan:
            self.replay.store(obs, action, reward, next_obs, done)

    def batch_replay(self):
        return self.replay.sample_batch(self.batch_size)

    def __len__(self):
        return self.replay.size


class RlTrainer():
    def __init__(self, env_name, logger=None) -> None:
        self.env_name = env_name
        self.env = gym.make(self.env_name)

        env_info = {'action_space': self.env.action_space,
                    'obs_space': self.env.observation_space}

        self.agent = Sac(env_info)

        self.logger = logger

        self.step_counter = 0
        self.update_every = 50
        self.batch_size = 250

        self.episode_counter = 0
        self.ep_reward = 0.
        self.ep_reward_ma = 0.
        self.ep_reward_ma_decay = 0.99

        self.exit_training = False

    def train(self):
        while self.step_counter < 1000:
            self.run_episode('random')

        while self.step_counter < int(1e7) and not self.exit_training:
            self.run_episode('train')

    def run_episode(self, mode):
        obs, *_ = self.env.reset()
        terminated = False
        while not terminated:
            if mode == 'train':
                # self.env

                a = self.agent.act(
                    torch.as_tensor(obs, dtype=torch.float32))
            elif mode == 'random':
                a = self.env.action_space.sample()
            elif mode == 'test':
                a = self.agent.act(torch.as_tensor(
                    obs, dtype=torch.float32), deterministic=True)

            next_obs, rew, terminal, timeout, info = self.env.step(a)

            self.ep_reward += rew

            self.agent.add_replay(obs, a, rew, next_obs, terminal)
            obs = next_obs

            if terminal or timeout:
                terminated = True

                self.ep_reward_ma = self.ep_reward_ma * self.ep_reward_ma_decay + \
                    (1-self.ep_reward_ma_decay) * self.ep_reward

                if self.logger is not None:
                    self.logger.experiment.add_scalar(
                        'ep_reward', self.ep_reward, self.episode_counter)

                self.ep_reward = 0.
                self.episode_counter += 1

            if mode == 'train' and self.step_counter % self.update_every == 0:

                for _ in range(self.update_every):
                    batch = self.agent.replay.sample_batch(self.batch_size)
                    self.agent.update(batch)

            self.step_counter += 1


if __name__ == '__main__':
    import pytorch_lightning.loggers.tensorboard as tb_logger
    trainer = RlTrainer('CartPole-v1', tb_logger.TensorBoardLogger('logs/'))
    trainer.train()

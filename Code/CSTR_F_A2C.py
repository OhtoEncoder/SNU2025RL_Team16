import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanDeVusseEnv(gym.Env):
    def __init__(self):
        super(VanDeVusseEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]),
                                                 high=np.array([10.0, 10.0]),
                                                 dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]), 
            high=np.array([1.0]), 
            dtype=np.float32
        )

        self.V = 100.0
        self.CA0 = 10.0
        self.k1 = 1.0
        self.k2 = 0.5
        self.k3 = 0.1
        self.dt = 0.1
        self.F_min = 1.0
        self.F_max = 20.0

        self.reset()

    def step(self, action):
        delta_F = float(action)
        self.F = np.clip(self.F + delta_F, self.F_min, self.F_max)

        CA, CB = self.state
        F = self.F

        dCA = (F / self.V) * (self.CA0 - CA) - self.k1 * CA - 2 * self.k3 * CA**2
        dCB = - (F / self.V) * CB + self.k1 * CA - self.k2 * CB

        CA = max(CA + dCA * self.dt, 0.0)
        CB = max(CB + dCB * self.dt, 0.0)

        self.state = np.array([CA, CB])
        reward = CB - 0.01 * F

        self.time += self.dt
        done = self.time >= 20.0

        return self.state.astype(np.float32), reward, done, {}

    def reset(self):
        self.state = np.array([5.0, 0.0])
        self.F = 10.0
        self.time = 0.0
        return self.state.astype(np.float32)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.base(x)
        mu = self.mu(x)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.v(x)


class A2CAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.hidden_dim = config['hidden_dim']
        self.discount_factor = config['gamma']
        self.lr_actor = config['lr_actor']
        self.lr_critic = config['lr_critic']
    
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic = Critic(self.state_dim, self.hidden_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)


config = {
    'num_episodes': 100,
    'hidden_dim': 256,
    'gamma': 0.99,
    'lr_actor': 3e-4,
    'lr_critic': 1e-3
}

env = VanDeVusseEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

agent = A2CAgent(state_dim, action_dim, config)

scores, episodes = [], []
best_score = -float('inf')

for ep in range(config['num_episodes']):
    s = env.reset()
    done = False
    ep_reward = 0

    while True:
        state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
        dist = agent.actor(state_tensor)
        value = agent.critic(state_tensor)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1, keepdim=True)
        clipped_action = torch.tanh(action) * action_bound
        a = clipped_action.cpu().squeeze(0).numpy()

        s2, r, done, _ = env.step(a)

        next_state_tensor = torch.FloatTensor(s2).unsqueeze(0).to(device)
        next_value = agent.critic(next_state_tensor) if not done else torch.tensor([[0.0]], device=device, dtype=torch.float32)

        reward_tensor = torch.tensor([[r]], device=device, dtype=torch.float32)
        target = reward_tensor + agent.discount_factor * next_value
        advantage = target - value

        actor_loss = -(log_prob * advantage.detach()).mean()
        critic_loss = F.mse_loss(value, target.detach())

        agent.actor_opt.zero_grad()
        actor_loss.backward()
        agent.actor_opt.step()

        agent.critic_opt.zero_grad()
        critic_loss.backward()
        agent.critic_opt.step()

        ep_reward += r
        s = s2

        if done:
            print(f"episode: {ep:3d} | score: {ep_reward:3.2f}")

            # Result plot
            scores.append(ep_reward)
            episodes.append(ep)
            plt.clf()
            plt.plot(episodes, scores, 'b')
            plt.xlabel("episode")
            plt.ylabel("score")
            plt.savefig("./A2C_F_graph.png")

            if ep_reward > best_score:
                best_score = ep_reward
                torch.save(agent.actor.state_dict(), "./A2C_F_actor.pth")
                torch.save(agent.critic.state_dict(), "./A2C_F_critic.pth")
                print(f"✔️ 새 최고 점수 {best_score:.2f}로 모델 저장됨.")
            
            break
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

agent.actor.load_state_dict(torch.load("./A2C_F_actor.pth"))
agent.actor.eval()
agent.critic.load_state_dict(torch.load("./A2C_F_critic.pth"))
agent.critic.eval()

# 환경 초기화
s = env.reset()
done = False

# 기록용 배열
time_list = [0.0]
F_list = [env.F]
CA_list = [env.state[0]]
CB_list = [env.state[1]]

while not done:
    state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = agent.actor(state_tensor)
        action = dist.sample()
        clipped_action = torch.tanh(action) * action_bound
    a = clipped_action.cpu().squeeze(0).numpy()

    s, r, done, _ = env.step(a)

    # 기록
    time_list.append(env.time)
    F_list.append(env.F)
    CA_list.append(env.state[0])
    CB_list.append(env.state[1])

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(time_list, F_list, label="Flow Rate F")
plt.ylabel("F")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_list, CA_list, label="CA", color='g')
plt.ylabel("CA")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_list, CB_list, label="CB", color='r')
plt.ylabel("CB")
plt.xlabel("Time")
plt.legend()

plt.tight_layout()
plt.savefig("A2C_F_trajectory.png")
plt.show()
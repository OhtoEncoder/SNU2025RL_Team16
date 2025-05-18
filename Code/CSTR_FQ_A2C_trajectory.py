import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanDeVusseEnvWithTemp(gym.Env):
    def __init__(self):
        super(VanDeVusseEnvWithTemp, self).__init__()

        # 상태: [C_A, C_B, T]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 300.0]),
            high=np.array([10.0, 10.0, 400.0]),
            dtype=np.float32
        )

        # 행동: ΔF, ΔQ
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1000.0]), 
            high=np.array([1.0, 1000.0]), 
            dtype=np.float32
        )

        self.V = 100.0
        self.CA0 = 10.0
        self.k1_base = 1.0
        self.k2_base = 0.5
        self.k3_base = 0.1
        self.Ea1 = 5000.0
        self.Ea2 = 6000.0
        self.Ea3 = 8000.0
        self.R = 8.314

        self.dt = 0.1
        self.F_min = 1.0
        self.F_max = 20.0

        self.T_in = 330.0
        self.Q = 0.0

        self.reset()

    def arrhenius(self, k0, Ea, T):
        return k0 * np.exp(-Ea / (self.R * T))

    def step(self, action):
        delta_F = float(action[0])
        delta_Q = float(action[1])

        self.F = np.clip(self.F + delta_F, self.F_min, self.F_max)
        self.Q = delta_Q

        CA, CB, T = self.state
        F = self.F

        k1 = self.arrhenius(self.k1_base, self.Ea1, T)
        k2 = self.arrhenius(self.k2_base, self.Ea2, T)
        k3 = self.arrhenius(self.k3_base, self.Ea3, T)

        dCA = (F / self.V) * (self.CA0 - CA) - k1 * CA - 2 * k3 * CA**2
        dCB = - (F / self.V) * CB + k1 * CA - k2 * CB

        Cp = 4.18 * 1000
        rho = 1000
        dT = (F / self.V) * (self.T_in - T) + self.Q / (self.V * rho * Cp)

        CA += dCA * self.dt
        CB += dCB * self.dt
        T += dT * self.dt

        CA = max(CA, 0.0)
        CB = max(CB, 0.0)
        T = np.clip(T, 300.0, 400.0)

        self.state = np.array([CA, CB, T])
        reward = CB - 0.01 * F - 0.0001 * abs(self.Q)

        self.time += self.dt
        done = self.time >= 20.0

        return self.state.astype(np.float32), reward, done, {}

    def reset(self):
        self.state = np.array([5.0, 0.0, 330.0])
        self.F = 10.0
        self.Q = 0.0
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

env = VanDeVusseEnvWithTemp()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

agent = A2CAgent(state_dim, action_dim, config)

agent.actor.load_state_dict(torch.load("./A2C_FQ_actor.pth"))
agent.critic.load_state_dict(torch.load("./A2C_FQ_critic.pth"))
agent.actor.eval()

# 환경 초기화
s = env.reset()
done = False

# 기록용 리스트 초기화
time_list = [0.0]
CA_list = [env.state[0]]
CB_list = [env.state[1]]
T_list = [env.state[2]]
F_list = [env.F]
Q_list = [env.Q]

action_bound = env.action_space.high  # shape = (2,)

while not done:
    state_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
    with torch.no_grad():
        dist = agent.actor(state_tensor)
        action = dist.sample()
        clipped_action = torch.tanh(action) * torch.from_numpy(action_bound).to(device)
    
    a = clipped_action.cpu().squeeze(0).numpy()  # shape: (2,)
    s, r, done, _ = env.step(a)

    # 값 기록
    time_list.append(env.time)
    CA_list.append(env.state[0])
    CB_list.append(env.state[1])
    T_list.append(env.state[2])
    F_list.append(env.F)
    Q_list.append(env.Q)

# 결과 시각화
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(time_list, CA_list, label="CA")
plt.ylabel("CA")
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(time_list, CB_list, label="CB", color='g')
plt.ylabel("CB")
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(time_list, T_list, label="T (K)", color='r')
plt.ylabel("Temperature")
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(time_list, F_list, label="Flow rate F", color='purple')
plt.ylabel("F")
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(time_list, Q_list, label="Heat Q", color='orange')
plt.ylabel("Q")
plt.xlabel("Time")
plt.legend()

plt.tight_layout()
plt.savefig("A2C_FQ_trajectory.png")
plt.show()
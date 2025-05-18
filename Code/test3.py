import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

class VanDeVusseEnvWithTemp(gym.Env):
    def __init__(self):
        super(VanDeVusseEnvWithTemp, self).__init__()

        # 상태: [C_A, C_B, T]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 300.0]),
            high=np.array([10.0, 10.0, 400.0]),
            dtype=np.float32
        )

        # 행동: ΔF (Discrete 5), ΔQ (Discrete 3) → PPO에서 다룰 수 있도록 MultiDiscrete 유지
        self.action_space = gym.spaces.MultiDiscrete([5, 3])
        self.F_delta_list = [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.Q_delta_list = [-1000.0, 0.0, 1000.0]

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
        f_idx, q_idx = action  # PPO는 action을 array로 반환함
        delta_F = self.F_delta_list[f_idx]
        delta_Q = self.Q_delta_list[q_idx]

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



# ----------------- PPO 에이전트 구성 ------------------
model = PPO(
    policy="MlpPolicy",
    env=VanDeVusseEnvWithTemp(),
    learning_rate=1e-3,
    gamma=0.99,
    verbose=1,
    batch_size=64,
    n_steps=2048,
    ent_coef=0.01,
)

# ----------------- 학습 ------------------
model.learn(total_timesteps=100_000)

# ----------------- 평가 ------------------
env_eval = VanDeVusseEnvWithTemp()
obs = env_eval.reset()
total_reward = 0
obs_list = []
F_list = []
Q_list = []

for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env_eval.step(action)
    total_reward += reward
    obs_list.append(obs[:2])  # CA, CB
    F_list.append(env_eval.F)
    Q_list.append(env_eval.Q)
    if done:
        break

print("총 reward:", total_reward)

# ----------------- 결과 시각화 ------------------
CA_list, CB_list = zip(*obs_list)
plt.plot(CA_list, label="C_A")
plt.plot(CB_list, label="C_B")
plt.plot(F_list, label="F (Flow Rate)")
plt.plot(Q_list, label="Q (Cooling)")
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("PPO Control of Van de Vusse Reactor")
plt.legend()
plt.grid(True)
plt.show()

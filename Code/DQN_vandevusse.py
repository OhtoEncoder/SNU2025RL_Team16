import os
import sys
import gym
from gym import spaces
import pygame
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cpu")

class VanDeVusseEnv(gym.Env):
    def __init__(self):
        super(VanDeVusseEnv, self).__init__()

        # 상태: C_A, C_B
        self.state_dim = 2
        self.action_dim = 3

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]),
                                            high=np.array([10.0, 10.0]),
                                            dtype=np.float32)

        # 행동: 유량 변화 ΔF ∈ {-2, -1, 0, +1, +2}
        self.action_space = spaces.Discrete(3)
        self.F_delta_list = [-1, 0, 1]

        # 고정 파라미터
        self.V = 10.0           # L
        self.CA0 = 3          # mol/L
        self.CB_target = 1.5
        self.k1 = 0.5           # 1/min
        self.k2 = 1            # 1/min
        self.k3 = 0.05           # L/mol·min

        # 시간 및 초기화
        self.dt = 1            # min
        self.F_min = 0.5         # L/min
        self.F_max = 5        # L/min

        self.reset()

    def step(self, action):
        # 유량 업데이트
        delta_F = self.F_delta_list[action]
        self.F = np.clip(self.F + delta_F, self.F_min, self.F_max)

        CA, CB = self.state
        F = self.F

        # 미분방정식 계산 (Euler method)
        dCA = (F / self.V) * (self.CA0 - CA) - self.k1 * CA - 2 * self.k3 * CA**2
        dCB = - (F / self.V) * CB + self.k1 * CA - self.k2 * CB

        CA += dCA * self.dt
        CB += dCB * self.dt

        CA = max(CA, 0.0)
        CB = max(CB, 0.0)

        self.state = np.array([CA, CB])


        # 보상 함수 (예시): CB 최대화, 유량은 적게
        # reward = 0

        # if action == 1: 
        #     reward += 0.1

        error = abs(self.CB_target - CB)
        reward = 1- error ** 2


        # if abs(error) < 0.1:
        #     reward += 100 - 0.01*F
        # else:
        #     reward += -(error**2) - 0.01*F

        # reward = CB - 0.01 * F

        # 종료 조건
        self.time += self.dt
        done = self.time >= 200 or CA < 0.01

        return self.state.astype(np.float32), reward, done

    def reset(self):
        self.state = np.array([2.0, 0.0])  # 초기 상태
        self.F = 2.5                     # 초기 유량
        self.time = 0.0
        return self.state.astype(np.float32)

class VanDeVusseEnv2(gym.Env):
    def __init__(self):
        super(VanDeVusseEnv2, self).__init__()
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]),
                                                 high=np.array([10.0, 10.0]),
                                                 dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.F_delta_list = [-2.0, -1.0, 0.0, 1.0, 2.0]

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
        delta_F = self.F_delta_list[action]
        self.F = np.clip(self.F + delta_F, self.F_min, self.F_max)

        CA, CB = self.state
        F = self.F

        dCA = (F / self.V) * (self.CA0 - CA) - self.k1 * CA - 2 * self.k3 * CA**2
        dCB = - (F / self.V) * CB + self.k1 * CA - self.k2 * CB

        CA += dCA * self.dt
        CB += dCB * self.dt
        CA = max(CA, 0.0)
        CB = max(CB, 0.0)

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

#%%
# DQN network structure
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc_out(x)
        return q
    
    
# DQN 
class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim 

        self.discount_factor = config['discount_factor']
        self.learning_rate = config['learning_rate']
        self.epsilon = config['epsilon']
        self.epsilon_decay = config['epsilon_decay']
        self.epsilon_min = config['epsilon_min']
        self.batch_size = config['batch_size']
        self.train_start = config['train_start']

        self.ReplayBuffer = deque(maxlen=5000)

        self.main_model = DQN(state_dim, action_dim, config['hidden_dim']).to(device)  # GPU로 이동
        self.target_model = DQN(state_dim, action_dim, config['hidden_dim']).to(device)  # GPU로 이동
        self.optimizer = optim.Adam(self.main_model.parameters(), lr=self.learning_rate)

        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.main_model.state_dict())

    def get_action(self, state, training=True):
        if training:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_dim)
            else:
                state = torch.FloatTensor(state).to(device)  
                q_value = self.main_model(state)
                return torch.argmax(q_value).item()
        else:
            state = torch.FloatTensor(state).to(device)  
            q_value = self.main_model(state)
            return torch.argmax(q_value).item() 

    def append_sample(self, state, action, reward, next_state, done):
        self.ReplayBuffer.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.ReplayBuffer, self.batch_size)

        states = torch.FloatTensor(np.array([sample[0] for sample in mini_batch])).to(device)  # FloatTensor for float type
        actions = torch.LongTensor(np.array([sample[1] for sample in mini_batch])).to(device)  # LongTensor for int type
        rewards = torch.FloatTensor(np.array([sample[2] for sample in mini_batch])).to(device)
        next_states = torch.FloatTensor(np.array([sample[3] for sample in mini_batch])).to(device)
        dones = torch.FloatTensor(np.array([sample[4] for sample in mini_batch])).to(device)

        main_q = self.main_model(states).squeeze(1)
        one_hot_action = torch.nn.functional.one_hot(actions, self.action_dim).float()
        main_q = torch.sum(one_hot_action * main_q, dim=1)

        target_q = self.target_model(next_states)
        target_q = target_q.detach()
        max_q = target_q.squeeze(1).max(1)[0]
        target_q = rewards + (1 - dones) * self.discount_factor * max_q
        
        loss = F.mse_loss(main_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    
    config = {'num_episodes': 100,
            #   'num_episodes': 10000,
              'hidden_dim': 24,
              'epsilon': 1.0,
#              'epsilon_decay': 0.999,
              'epsilon_decay': 0.995,
              'epsilon_min': 0.05,
              'batch_size': 64,
              'train_start': 1000,
              'discount_factor': 0.99,
              'learning_rate': 0.01
               }
    
    env = VanDeVusseEnv2()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, config)
    scores, timesteps, episodes = [], [], []
    score_avg = 0
    best_score = 0

    for e in range(config['num_episodes']):
        steps = 0
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_dim])
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_dim])
            
            score += reward


            agent.append_sample(state, action, reward, next_state, done)
            
            if len(agent.ReplayBuffer) >= agent.train_start:
                agent.train_model()

            state = next_state

            if done:
                agent.update_target_model()
                print(f"episode: {e:3d} | score: {score:3.2f} | memory length: {len(agent.ReplayBuffer):4d} | epsilon: {agent.epsilon:.4f}")

                # Result plot
                scores.append(score)
                episodes.append(e)
                timesteps.append(env.time)
                plt.plot(episodes, scores, 'b')
                plt.xlabel("episode")
                plt.ylabel("score")
                plt.savefig("./DQN_graph.png")

                # 학습 루프 안에서
                if score > 300:
                    average_score = 0
                    for i in range(5):
                        score = 0
                        done = False
                        state = env.reset()
                        state = np.reshape(state, [1, state_dim])

                        while not done:
                            action = agent.get_action(state, False)
                            next_state, reward, done, _ = env.step(action)
                            next_state = np.reshape(next_state, [1, state_dim])
                            state = next_state
                            score += reward

                        average_score += score
                    average_score /= 5  # 평균 내기

                    # ✅ 최대 평균 점수일 때만 저장
                    if average_score > best_score:
                        best_score = average_score
                        torch.save(agent.main_model.state_dict(), "./DQN_model.pth")
                        print(f"✔️ 새 최고 평균 점수 {average_score:.2f}로 모델 저장됨.")

#%%
agent = DQNAgent(state_dim, action_dim, config)
agent.main_model.load_state_dict(torch.load('DQN_model.pth'))
#%%
env = VanDeVusseEnv2()
agent.epsilon=0.01
done = False
score = 0
state = env.reset()
state = np.reshape(state, [1, state_dim])

while not done:
    action = agent.get_action(state, False)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_dim])
    
    score += reward
    state = next_state
    
    if done:
        print(f"score: {score:3.2f}")

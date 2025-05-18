# SNU2025RL_Team16

## 📎 Source and Setup

	•	Environment and custom reaction dynamics are located in: pc-gym (modified locally)
	•	See requirements.txt and env.yml for virtual environment setup.


## 📌 Repository Highlights
 
	•	🔁 Supports RL Control on Chemical CSTR Reactor
	•	🔬 Includes Complex Reaction Kinetics (Van de Vusse)
	•	🧪 Visualizes learning curve and state trajectory for interpretability


## 📁 Code Folder Overview


This folder contains training scripts, saved models, and result visualizations for RL-based control of Continuous Stirred Tank Reactor (CSTR) systems, including the Van de Vusse reaction.


We implemented and compared multiple RL algorithms (e.g., DQN, A2C) across different control setups:

	•	F: Feed flow control only
	•	FQ: Feed + Heat (Q) control (multi-action)

Each file is named according to its control type, algorithm, and purpose, as explained below.

## 🗂️ File Naming Convention

| Prefix / Pattern | Meaning |
|------------------|---------|
| `CSTR_` | Environment: **Continuous Stirred Tank Reactor (CSTR)** |
| `F` | Control variable: **Feed flow only** |
| `FQ` | Control variables: **Feed flow + Heat input (Q)** |
| `A2C` | Algorithm: **Advantage Actor-Critic** |
| `DQN` | Algorithm: **Deep Q-Network** |
| `_trajectory.py` | Script to **plot or analyze** the trajectory of a trained model |
| `_graph.png` | Training performance **graph** (e.g., reward vs. episode) |
| `_actor.pth`, `_critic.pth` | Saved **model weights** (PyTorch) for actor and critic networks |

📝 Example:
- `CSTR_F_A2C.py`: A2C training on CSTR using feed flow only  
- `CSTR_FQ_A2C_trajectory.py`: Trajectory visualization for feed+heat control  
- `A2C_F_graph.png`: Training graph for feed-only A2C model  

## 📌 File Descriptions

| File Name | Description |
|-----------|-------------|
| `CSTR_F_A2C.py` | A2C training script on CSTR with **Feed-only** control |
| `CSTR_F_A2C_trajectory.py` | Plot or visualize A2C trajectories from Feed-only experiment |
| `CSTR_FQ_A2C.py` | A2C training script on CSTR with **Feed + Heat** control |
| `CSTR_FQ_A2C_trajectory.py` | Plot or visualize A2C trajectories from Feed+Q experiment |
| `DQN_vandevusse.py` | DQN applied to **Van de Vusse reaction** environment |
| `A2C_F_actor.pth`, `A2C_F_critic.pth` | Trained A2C actor & critic weights (Feed-only) |
| `A2C_FQ_actor.pth`, `A2C_FQ_critic.pth` | Trained A2C actor & critic weights (Feed + Heat) |
| `A2C_F_graph.png`, `A2C_F_trajectory.png` | Feed-only training results and trajectory visualizations |
| `A2C_FQ_graph.png`, `A2C_FQ_trajectory.png` | Feed + Heat training results and trajectory visualizations |

## 📘 Pseudocode of [DQN_vandevusse.py]

**Initialize** CSTR environment with state `[C_A, C_B]`  
**Define** action space: ΔF ∈ {-2, -1, 0, +1, +2}  
**Define** reward: `r = C_B − 0.01 × F`

---

**Function** `Step(state, action)`  
- **Update** flow rate `F` using selected `ΔF`  
- **Compute** concentration changes via Euler integration:  
  - `dC_A = feed − reaction loss`  
  - `dC_B = reaction gain − output loss`  
- **Apply** time step to update `[C_A, C_B]`  
- **Calculate** reward and check termination  
- **Return** `next_state`, `reward`, `done_flag`

---

**Function** `Reset()`  
- **Initialize** state and flow rate  
- **Reset** simulation time  
- **Return** `initial_state`

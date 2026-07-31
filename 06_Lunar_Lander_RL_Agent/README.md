# Project 06 — Lunar Lander Reinforcement Learning Agent (Dueling DQN)

**Author:** Rudhra Sitholey  
**Registration No:** 23BCY10296 | **Application No:** IN26012560  
**Email:** rudhra.23bcy10296@vitbhopal.ac.in  

---

## 📌 Project Overview

A complete Dueling Deep Q-Network (Dueling DQN) reinforcement learning agent solving the **LunarLander** Box2D control environment using PyTorch, Gymnasium, and a custom Pygame visual renderer.

- **Environment**: `LunarLander-v3` / `LunarLander-v2` (Gymnasium Box2D)
- **Observation Space**: 8 continuous state values (position $(x, y)$, velocities $(v_x, v_y)$, orientation angle $\theta$, angular velocity $\omega$, and two leg contact booleans)
- **Action Space**: 4 discrete actions (0: Do Nothing, 1: Fire Left Engine, 2: Fire Main Engine, 3: Fire Right Engine)
- **Architecture**: Dueling DQN (decoupled State-Value $V(s)$ and Advantage $A(s, a)$ streams) with Double Q-Learning target updates and soft network synchronization ($\tau = 0.001$).
- **Custom Visual Style**: Pygame simulation window featuring a flat moon surface, 2 blue landing pad markers, white spaceship body, and red engine thruster sparks.

---

## 📁 Repository Structure

```
06_Lunar_Lander_RL_Agent/
├── model.py              # Dueling Q-Network architecture
├── agent.py              # LunarLanderAgent with Soft Updates & Double Q-Learning
├── custom_lunar.py       # Custom styled Pygame environment renderer
├── train.py              # Training loop with convergence monitoring & plotting
├── evaluate.py           # Evaluation script for testing policy
├── visualize.py          # Live native Pygame simulation window runner
├── lunar_lander_dqn.pth  # Saved trained checkpoint weights
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
cd 06_Lunar_Lander_RL_Agent
pip install -r requirements.txt
```

### 2. Train the Agent

To train the Dueling DQN agent on LunarLander:

```bash
python train.py
```

### 3. Launch Live Visual Simulation Window

To open the **custom Pygame visual simulation window** showing the trained lander flying and landing live on screen:

```bash
python visualize.py
```

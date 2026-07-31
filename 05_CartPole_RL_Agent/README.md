# Project 05 — CartPole Reinforcement Learning Agent (PyTorch & Gymnasium)

**Author:** Rudhra Sitholey  
**Registration No:** 23BCY10296 | **Application No:** IN26012560  
**Email:** rudhra.23bcy10296@vitbhopal.ac.in  

---

## 📌 Project Overview

A complete Deep Q-Network (DQN) reinforcement learning implementation to solve the classic **CartPole-v1** control environment using PyTorch, Gymnasium, and Pygame.

- **Environment**: `CartPole-v1` (Gymnasium)
- **Observation Space**: 4 continuous variables (Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity)
- **Action Space**: 2 discrete actions (0: Move Left, 1: Move Right)
- **Algorithm**: Deep Q-Learning (DQN) with Experience Replay Buffer and Target Q-Network

---

## 📁 Repository Structure

```
05_CartPole_RL_Agent/
├── model.py           # PyTorch Q-Network architecture
├── agent.py           # DQNAgent and Experience Replay Buffer
├── train.py           # Main training script with rewards plotting
├── evaluate.py        # Policy evaluation script
├── visualize.py       # Live native Pygame simulation window runner
├── cartpole_dqn.pth   # Saved trained model weights checkpoint
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
cd 05_CartPole_RL_Agent
pip install -r requirements.txt
```

### 2. Train the Agent

To train the DQN agent on CartPole and generate performance graphs:

```bash
python train.py
```

### 3. Launch Live Visual Simulation Window

To open the **native Pygame visual simulation window** showing the trained agent balancing the pole live on screen:

```bash
python visualize.py
```

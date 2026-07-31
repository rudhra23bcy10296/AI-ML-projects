import os
import time
import gymnasium as gym
import torch
import numpy as np
from agent import DQNAgent

def run_cartpole_simulation(episodes: int = 5):
    print("=" * 60)
    print(" 🎬 Launching Live CartPole-v1 Pygame Window Simulation")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "cartpole_dqn.pth")
    
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"[+] Loaded trained model weights from: {model_path}")
    else:
        print(f"[!] Model weights '{model_path}' not found! Running untrained agent policy.")
        
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
            step += 1
            time.sleep(0.02) # Control frame rate for smooth visual viewing
            
        print(f"Episode {ep:2d}/{episodes} Finished | Total Steps / Score: {total_reward:.1f}")
        time.sleep(0.5)
        
    env.close()
    print("=" * 60)
    print("[+] Simulation complete. Window closed.")

if __name__ == "__main__":
    run_cartpole_simulation(episodes=5)

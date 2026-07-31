import os
import time
import torch
import numpy as np
from agent import LunarLanderAgent
from custom_lunar import CustomStyledLunarLander

def run_lunar_simulation(episodes: int = 5):
    print("=" * 60)
    print(" 🎬 Launching Custom Styled LunarLander Pygame Simulation Window")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "lunar_lander_dqn.pth")
    
    # Custom styled LunarLander environment
    env = CustomStyledLunarLander(render_mode="human")
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = LunarLanderAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
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
            time.sleep(0.015) # Control frame rate for smooth visual viewing
            
        print(f"Episode {ep:2d}/{episodes} Finished | Total Steps: {step:3d} | Score: {total_reward:6.2f}")
        time.sleep(0.5)
        
    env.close()
    print("=" * 60)
    print("[+] Custom visual simulation complete. Window closed.")

if __name__ == "__main__":
    run_lunar_simulation(episodes=5)

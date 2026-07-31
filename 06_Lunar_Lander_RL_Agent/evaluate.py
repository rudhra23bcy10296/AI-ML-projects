import os
import torch
import numpy as np
from agent import LunarLanderAgent
from custom_lunar import CustomStyledLunarLander

def evaluate_lunar_lander(
    model_path: str,
    num_episodes: int = 5,
    render: bool = True
):
    print("=" * 60)
    print(" Evaluating Custom Styled LunarLander Agent")
    print("=" * 60)
    
    render_mode = "human" if render else None
    env = CustomStyledLunarLander(render_mode=render_mode)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = LunarLanderAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"[+] Successfully loaded checkpoint from {model_path}")
    else:
        print(f"[!] Warning: Model path '{model_path}' not found! Running untrained agent.")
        
    rewards = []
    
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            steps += 1
            
        rewards.append(episode_reward)
        print(f"Eval Episode {episode:2d}/{num_episodes} | Steps: {steps:3d} | Reward: {episode_reward:6.2f}")
        
    env.close()
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print("-" * 60)
    print(f"Evaluation Summary ({num_episodes} episodes):")
    print(f"  Mean Reward : {mean_reward:.2f} +/- {std_reward:.2f}")
    print("=" * 60)
    return rewards

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "lunar_lander_dqn.pth")
    evaluate_lunar_lander(model_path, num_episodes=5, render=True)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from agent import LunarLanderAgent
from custom_lunar import CustomStyledLunarLander

def train_lunar_lander(
    episodes: int = 200,
    save_path: str = "lunar_lander_dqn.pth",
    target_reward: float = 200.0
):
    print("=" * 60)
    print(" Starting Custom Styled LunarLander Dueling DQN Training Routine")
    print("=" * 60)
    
    env = CustomStyledLunarLander()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device}")
    agent = LunarLanderAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    rewards_history = []
    recent_rewards = []
    
    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, float(done))
            agent.update()
            
            state = next_state
            episode_reward += reward
            
        agent.decay_epsilon()
        rewards_history.append(episode_reward)
        recent_rewards.append(episode_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
            
        avg_reward = np.mean(recent_rewards)
        
        if episode % 10 == 0 or avg_reward >= target_reward:
            print(f"Episode {episode:3d}/{episodes} | Avg Reward (last 100): {avg_reward:6.2f} | Ep Reward: {episode_reward:6.2f} | Epsilon: {agent.epsilon:.4f}")
            
        if avg_reward >= target_reward:
            print(f"\n[+] Environment Solved! Average reward ({avg_reward:.2f}) exceeded target {target_reward} at episode {episode}!")
            break
            
    env.close()
    
    # Save trained model weights
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    agent.save(save_path)
    print(f"\n[+] Saved trained model checkpoint to: {save_path}")
    
    # Plot reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label="Episode Reward", alpha=0.5, color="crimson")
    if len(rewards_history) >= 20:
        ma = np.convolve(rewards_history, np.ones(20)/20, mode='valid')
        plt.plot(range(19, len(rewards_history)), ma, label="20-Episode Moving Avg", color="darkred", linewidth=2)
        
    plt.axhline(y=200, color='blue', linestyle='--', label='Solved Threshold (200)')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Custom LunarLander Dueling DQN Training Progress")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plot_path = save_path.replace(".pth", "_rewards.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Saved training rewards plot to: {plot_path}")
    
    return agent, rewards_history

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(script_dir, "lunar_lander_dqn.pth")
    train_lunar_lander(episodes=200, save_path=model_save_path)

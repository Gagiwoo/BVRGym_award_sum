import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from jsb_gym import BVR2v2Env  # 2 vs 2 BVR 환경
from happo import HAPPOAgent  # HAPPO 기반 MARL


# 환경 초기화
env = BVR2v2Env()
n_agents = env.n_agents

# HAPPO 설정
agents = [HAPPOAgent(env.observation_space[i].shape, env.action_space[i].n) for i in range(n_agents)]
optimizer = optim.Adam([p for agent in agents for p in agent.parameters()], lr=3e-4)

episodes = 1000
batch_size = 64
episode_rewards = []
window_size = 50
moving_avg_rewards = []

for episode in range(episodes):
    states = env.reset()
    done = [False] * n_agents
    total_reward = 0
    steps = 0
    while not any(done):
        actions = [agent.select_action(torch.tensor(state, dtype=torch.float32)) for agent, state in zip(agents, states)]
        next_states, rewards, done, _ = env.step(actions)

        for i, agent in enumerate(agents):
            agent.store_transition(states[i], actions[i], rewards[i], next_states[i], done[i])

        states = next_states
        total_reward += sum(rewards)
        steps += 1

    # 학습 진행
    if episode % batch_size == 0:
        for agent in agents:
            agent.learn()

    episode_rewards.append(total_reward / n_agents)
    moving_avg_rewards.append(np.mean(episode_rewards[-window_size:]))

    if episode % 50 == 0:
        print(f"Episode {episode}: Avg Reward = {moving_avg_rewards[-1]:.2f}")

# 학습 보상 그래프
plt.figure(figsize=(10,5))
plt.plot(episode_rewards, label="Episode Rewards", alpha=0.5)
plt.plot(moving_avg_rewards, label="Moving Avg (50 episodes)", color='red')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("BVR 2v2 Multi-Agent Learning Performance")
plt.legend()
plt.show()

# 모델 저장
for i, agent in enumerate(agents):
    torch.save(agent.state_dict(), f"happo_agent_{i}.pth")
print("Model saved!")

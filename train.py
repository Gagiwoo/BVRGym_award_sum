from envs.bvr_env import BVR_EvadeEnv
from stable_baselines3 import PPO

env = BVR_EvadeEnv()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./runs")
model.learn(total_timesteps=100_000)
model.save("ppo_bvr_evade")
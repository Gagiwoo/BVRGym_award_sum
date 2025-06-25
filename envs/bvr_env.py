import gym
import numpy as np
from jsb_gym.environments import evasive
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym
from jsb_gym.environments.config import BVRGym_PPO1

class BVR_EvadeEnv(gym.Env):
    def __init__(self):
        self.env = evasive.Evasive(BVRGym_PPO1, {}, aim_evs_BVRGym, f16_evs_BVRGym)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.env.observation_space, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def reset(self):
        obs_block = self.env.reset(rand_state_f16=True, rand_state_aim=True)
        return obs_block['aim1']  # 상태 벡터

    def step(self, action):
        obs_block, reward, done, info = self.env.step(action, action_type=0)
        return obs_block['aim1'], reward, done, info

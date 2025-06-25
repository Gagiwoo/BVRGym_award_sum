import gym
import jsb_gym

env = gym.make("jsb-v0")
obs = env.reset()

print("환경이 잘 초기화되었습니다!")
print("초기 관측값:", obs)
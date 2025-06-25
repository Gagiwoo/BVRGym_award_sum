# mainBVRGym_MultiCore.py - 환경 선택 기능 추가 완료

import argparse, time
from jsb_gym.environments import evasive, bvrdog
import numpy as np
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
import torch
import multiprocessing
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.environments.config import BVRGym_PPO1, BVRGym_PPO2, BVRGym_PPODog
from jsb_gym.environments.evasive import get_tb_obs_evasive  # ← 이거 추가!
from numpy.random import seed
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym, aim_dog_BVRGym, f16_dog_BVRGym

def init_pool():
    seed()

class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

def runPPO(args):
    if args['track'] == 'M1':
        from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
        env = evasive.Evasive(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M1.pth'
        state_scale = 1
    elif args['track'] == 'M2':
        from jsb_gym.RL.config.ppo_evs_PPO2 import conf_ppo
        env = evasive.Evasive(BVRGym_PPO2, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M2.pth'
        state_scale = 2
    elif args['track'] == 'Dog':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save = 'jsb_gym/logs/RL/Dog/'
        state_scale = 1
    elif args['track'] == 'DogR':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save = 'jsb_gym/logs/RL/DogR.pth'
        state_scale = 1

    method = args.get('method', 'baseline')
    conf_ppo.setdefault('general', {})  # 👈 이 줄로 general 블록 만들기
    conf_ppo['general']['method'] = method  # 👈 method 값 넣기

    if method == 'baseline':
        conf_ppo['normalize_rewards'] = False
        conf_ppo['lam_a'] = 0.0
        conf_ppo['entropy_weight'] = 0.01
    elif method == 'shaping':
        conf_ppo['normalize_rewards'] = True
    elif method == 'entropy':
        conf_ppo['entropy_weight'] = 0.05
    elif method == 'smooth':
        conf_ppo['lam_a'] = 0.1

    writer = SummaryWriter(f"runs/{args['track']}_{args['method']}")
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    memory = Memory()
    ppo = PPO(state_dim * state_scale, action_dim, conf_ppo, use_gpu=False)
    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)

    for i_episode in range(1, args['Eps']+1):
        ppo_policy = ppo.policy.state_dict()
        ppo_policy_old = ppo.policy_old.state_dict()
        input_data = [(args, ppo_policy, ppo_policy_old, conf_ppo, state_scale) for _ in range(args['cpu_cores'])]
        running_rewards = []
        tb_obs = []

        results = pool.map(train, input_data)
        for idx, tmp in enumerate(results):
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])
            tb_obs.append(tmp[6])

        ppo.set_device(use_gpu=True)
        ppo.update(memory, to_tensor=True, use_gpu=True)
        memory.clear_memory()
        ppo.set_device(use_gpu=False)
        torch.cuda.empty_cache()

        writer.add_scalar("running_rewards", sum(running_rewards)/len(running_rewards), i_episode)
        writer.add_scalar("policy_entropy", ppo.get_entropy(), i_episode)

        tb_obs0 = None
        for i in tb_obs:
            if tb_obs0 is None:
                tb_obs0 = i
            else:
                for key in tb_obs0:
                    tb_obs0[key] += i[key]

        if tb_obs0:
            nr = len(tb_obs)
            for key in tb_obs0:
                tb_obs0[key] = tb_obs0[key] / nr
                writer.add_scalar(key, tb_obs0[key], i_episode)

        # ✅ 보상 구성 요소 기록
        if hasattr(env, 'last_reward_components'):
            for k, v in env.last_reward_components.items():
                writer.add_scalar(k, v, i_episode)

        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), torch_save + f'Dog{i_episode}.pth')

    pool.close()
    pool.join()

def train(args):
    from jsb_gym.environments import evasive, bvrdog
    if args[0]['track'] == 'M1':
        env = evasive.Evasive(BVRGym_PPO1, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif args[0]['track'] == 'M2':
        env = evasive.Evasive(BVRGym_PPO2, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif args[0]['track'] == 'Dog':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)
    elif args[0]['track'] == 'DogR':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)

    maneuver = Maneuvers.Evasive
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim * args[4], action_dim, args[3], use_gpu=False)

    ppo.policy.load_state_dict(args[1])
    ppo.policy_old.load_state_dict(args[2])
    ppo.policy.eval()
    ppo.policy_old.eval()

    running_reward = 0.0
    for _ in range(args[0]['eps']):
        action = np.zeros(3)
        if args[0]['track'] == 'M1':
            state_block = env.reset(True, True)
            state = state_block['aim1']
            action[2] = 1
        elif args[0]['track'] == 'M2':
            state_block = env.reset(True, True)
            state = np.concatenate((state_block['aim1'][0], state_block['aim2'][0]))
            action[2] = 1
        elif args[0]['track'] in ['Dog', 'DogR']:
            state = env.reset()
            action[2] = 0.0

        done = False
        while not done:
            act = ppo.select_action(state, memory)
            action[0], action[1] = act[0], act[1]

            if args[0]['track'] == 'M1':
                state_block, reward, done, _ = env.step(action, action_type=maneuver.value)
                state = state_block['aim1']
            elif args[0]['track'] == 'M2':
                state_block, reward, done, _ = env.step(action, action_type=maneuver.value)
                state = np.concatenate((state_block['aim1'], state_block['aim2']))
            elif args[0]['track'] == 'Dog':
                state, reward, done, _ = env.step(action, action_type=maneuver.value, blue_armed=True, red_armed=True)
            elif args[0]['track'] == 'DogR':
                state, reward, done, _ = env.step(action, action_type=maneuver.value, blue_armed=False, red_armed=True)

            # 💡 shaping reward 적용
            shaping_r = 0
            if args[0].get("shaping_w", 0.0) > 0:
                shaping_r += env.f16.get_v_down(scaled=True)  # 예시: 하강속도 shaping
                reward += shaping_r * args[0]["shaping_w"]

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

        running_reward += reward

    running_reward /= args[0]['eps']

    tb_obs = {}
    if args[0]['track'] in ['Dog', 'DogR']:
        tb_obs = get_tb_obs_dog(env)
    elif args[0]['track'] in ['M1', 'M2']:  # ← 이 조건 추가!
        tb_obs = get_tb_obs_evasive(env)

    actions = [i.detach().numpy() for i in memory.actions]
    states = [i.detach().numpy() for i in memory.states]
    logprobs = [i.detach().numpy() for i in memory.logprobs]
    rewards = [i for i in memory.rewards]
    is_terminals = [i for i in memory.is_terminals]

    return [actions, states, logprobs, rewards, is_terminals, running_reward, tb_obs]

def get_tb_obs_dog(env):
    tb_obs = {}
    tb_obs['Blue_ground'] = env.reward_f16_hit_ground
    tb_obs['Red_ground'] = env.reward_f16r_hit_ground
    tb_obs['maxTime'] = env.reward_max_time
    tb_obs['Blue_alive'] = env.f16_alive
    tb_obs['Red_alive'] = env.f16r_alive
    return tb_obs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type=str, default='M1')
    parser.add_argument("-cpus", "--cpu_cores", type=int, default=4)
    parser.add_argument("-Eps", "--Eps", type=int, default=1000)
    parser.add_argument("-eps", "--eps", type=int, default=3)
    parser.add_argument("-w", "--shaping_w", type=float, default=0.0, help="shaping reward weight")
    parser.add_argument("-m", "--method", type=str, default='baseline',
                    help="사용할 기법: baseline | method1 | method2 | method3 | method4")
    args = vars(parser.parse_args())

    runPPO(args)

def get_tb_obs_evasive(env):
    return {
        'F16_dead': env.reward_f16_dead,
        'F16_ground': env.reward_f16_hit_ground,
        'AIM_ground': env.reward_aim_hit_ground,
        'Lost_all': env.reward_all_lost,
        'Max_time': env.reward_max_time
    }
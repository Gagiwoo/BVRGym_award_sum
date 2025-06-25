import argparse, time
from jsb_gym.environments import evasive, bvrdog
import numpy as np
from enum import Enum
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import torch
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.environments.config import BVRGym_PPO1, BVRGym_PPO2, BVRGym_PPODog
from numpy.random import seed
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym, aim_dog_BVRGym, f16_dog_BVRGym

def init_pool():
    seed()

class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

def logger_process(log_queue, log_dir):
    writer = SummaryWriter(log_dir)
    while True:
        data = log_queue.get()
        if data == 'TERMINATE':
            break
        tag, value, step = data
        writer.add_scalar(tag, value, step)
    writer.close()

def runPPO(args):
    track = args['track']
    if track == 'M1':
        from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
        env = evasive.Evasive(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M1.pth'
        state_scale = 1
    elif track == 'M2':
        from jsb_gym.RL.config.ppo_evs_PPO2 import conf_ppo
        env = evasive.Evasive(BVRGym_PPO2, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save = 'jsb_gym/logs/RL/M2.pth'
        state_scale = 2
    elif track == 'Dog':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save = 'jsb_gym/logs/RL/Dog/'
        state_scale = 1
    elif track == 'DogR':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save = 'jsb_gym/logs/RL/DogR.pth'
        state_scale = 1

    # Logger 프로세스 생성
    manager = multiprocessing.Manager()
    log_queue = manager.Queue()
    logger = multiprocessing.Process(target=logger_process, args=(log_queue, 'runs/' + track))
    logger.start()

    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    memory = Memory()
    ppo = PPO(state_dim * state_scale, action_dim, conf_ppo, use_gpu=False)
    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)

    for i_episode in range(1, args['Eps']+1):
        ppo_policy = ppo.policy.state_dict()
        ppo_policy_old = ppo.policy_old.state_dict()
        input_data = [(
            dict(args),  # args 복사본 생성
            ppo_policy,
            ppo_policy_old,
            conf_ppo,
            state_scale
        ) for _ in range(args['cpu_cores'])]
        results = pool.map(train, input_data)

        memory.clear_memory()
        running_rewards = []
        tb_obs = []

        for tmp in results:
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])
            tb_obs.append(tmp[6])

        ppo.set_device(use_gpu=True)
        ppo.update(memory, to_tensor=True, use_gpu=True)
        ppo.set_device(use_gpu=False)
        torch.cuda.empty_cache()

        # 평균 reward 기록
        avg_reward = sum(running_rewards) / len(running_rewards)
        log_queue.put(("running_rewards", avg_reward, i_episode))

        # 기타 관측값
        tb_obs0 = tb_obs[0]
        for obs in tb_obs[1:]:
            for key in obs:
                tb_obs0[key] += obs[key]
        for key in tb_obs0:
            tb_obs0[key] /= len(tb_obs)
            log_queue.put((key, tb_obs0[key], i_episode))

        # 모델 저장
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), torch_save + str(i_episode) + '.pth')

    pool.close()
    pool.join()
    log_queue.put("TERMINATE")
    logger.join()

def train(args):
    track = args[0]['track']
    if track == 'M1':
        env = evasive.Evasive(BVRGym_PPO1, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif track == 'M2':
        env = evasive.Evasive(BVRGym_PPO2, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    elif track == 'Dog':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)
    elif track == 'DogR':
        env = bvrdog.BVRDog(BVRGym_PPODog, args[0], aim_dog_BVRGym, f16_dog_BVRGym)

    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim * args[4], action_dim, args[3], use_gpu=False)

    ppo.policy.load_state_dict(args[1])
    ppo.policy_old.load_state_dict(args[2])

    running_reward = 0.0
    for _ in range(args[0]['eps']):
        done = False
        action = np.zeros(3)
        if track in ['M1', 'M2']:
            state_block = env.reset(True, True)
            if track == 'M1':
                state = state_block['aim1']
            else:
                state = np.concatenate((state_block['aim1'][0], state_block['aim2'][0]))
            action[2] = 1
        else:
            state = env.reset()
            action[2] = 0.0

        while not done:
            act = ppo.select_action(state, memory)
            action[0], action[1] = act[0], act[1]
            if track == 'M1':
                state_block, reward, done, _ = env.step(action, action_type=0)
                state = state_block['aim1']
            elif track == 'M2':
                state_block, reward, done, _ = env.step(action, action_type=0)
                state = np.concatenate((state_block['aim1'], state_block['aim2']))
            else:
                state, reward, done, _ = env.step(action, action_type=0)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

        running_reward += reward

    avg_reward = running_reward / args[0]['eps']
    tb_obs = get_tb_obs_dog(env) if 'Dog' in track else {}

    # 🔽 여기에 추가
    actions = [i.detach().cpu().numpy() for i in memory.actions]
    states = [i.detach().cpu().numpy() for i in memory.states]
    logprobs = [i.detach().cpu().numpy() for i in memory.logprobs]

    return [actions, states, logprobs, memory.rewards, memory.is_terminals, avg_reward, tb_obs]

def get_tb_obs_dog(env):
    return {
        'F16_dead': int(not env.f16_alive),
        'AIM_ground': float(env.aim_block['aim1'].target_hit),
        'loss_mse': getattr(env, 'loss_mse', 0)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vizualize", action='store_true', help="Render in FG")
    parser.add_argument("-track", "--track", type=str, help="Tracks: M1, M2, Dog, DogR", default='M1')
    parser.add_argument("-cpus", "--cpu_cores", type=int, default=4)
    parser.add_argument("-Eps", "--Eps", type=int, default=1000)
    parser.add_argument("-eps", "--eps", type=int, default=3)
    args = vars(parser.parse_args())
    runPPO(args)

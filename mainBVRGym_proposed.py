# mainBVRGym_MultiCore.py - 환경 선택 기능 추가 완료
# 실행 시 --mode baseline1|baseline2|baseline3|proposed 인자로 실험 환경 지정 가능

import argparse, time, torch, multiprocessing
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from numpy.random import seed
from enum import Enum
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.environments.config import BVRGym_PPO1
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym

# 다양한 버전의 환경 불러오기
import jsb_gym.environments.evasive_b1 as ev_b1
import jsb_gym.environments.evasive_b2 as ev_b2
import jsb_gym.environments.evasive_b3 as ev_b3
import jsb_gym.environments.evasive_proposed as ev_proposed

# 실험 환경 선택 함수
ENV_MAP = {
    'baseline1': ev_b1.Evasive,
    'baseline2': ev_b2.Evasive,
    'baseline3': ev_b3.Evasive,
    'proposed': ev_proposed.Evasive
}

def select_env(mode):
    return ENV_MAP.get(mode, ev_proposed.Evasive)

def logger_process(log_queue, log_dir):
    writer = SummaryWriter(log_dir)
    while True:
        data = log_queue.get()
        if data == 'TERMINATE':
            break
        tag, value, step = data
        writer.add_scalar(tag, value, step)
    writer.close()

def init_pool():
    seed()

def train(args):
    env_class = select_env(args[0]['mode'])
    env = env_class(BVRGym_PPO1, args[0], aim_evs_BVRGym, f16_evs_BVRGym)
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim * args[4], action_dim, args[3], use_gpu=False)
    ppo.policy.load_state_dict(args[1])
    ppo.policy_old.load_state_dict(args[2])

    running_reward = 0.0
    for _ in range(args[0]['eps']):
        done = False
        state_block = env.reset(True, True)
        state = state_block['aim1']
        action = np.zeros(3)
        action[2] = 1
        while not done:
            act = ppo.select_action(state, memory)
            action[0], action[1] = act[0], act[1]
            state_block, reward, done, _ = env.step(action, action_type=0)
            state = state_block['aim1']
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
        running_reward += reward

    avg_reward = running_reward / args[0]['eps']
    return [memory.actions, memory.states, memory.logprobs, memory.rewards, memory.is_terminals, avg_reward, {}]

def runPPO(args):
    env_class = select_env(args['mode'])
    env = env_class(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)
    torch_save = f'jsb_gym/logs/RL/{args["track"]}_{args["mode"]}.pth'

    manager = multiprocessing.Manager()
    log_queue = manager.Queue()
    logger = multiprocessing.Process(target=logger_process, args=(log_queue, 'runs/' + args['track']))
    logger.start()

    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    memory = Memory()
    from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
    ppo = PPO(state_dim, action_dim, conf_ppo, use_gpu=False)
    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)

    for i_episode in range(1, args['Eps'] + 1):
        ppo_policy = ppo.policy.state_dict()
        ppo_policy_old = ppo.policy_old.state_dict()
        input_data = [(
            dict(args), ppo_policy, ppo_policy_old, conf_ppo, 1
        ) for _ in range(args['cpu_cores'])]

        results = pool.map(train, input_data)

        memory.clear_memory()
        running_rewards = []
        for tmp in results:
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])

        ppo.set_device(use_gpu=True)
        ppo.update(memory, to_tensor=True, use_gpu=True)
        ppo.set_device(use_gpu=False)

        avg_reward = sum(running_rewards) / len(running_rewards)
        log_queue.put(("running_rewards", avg_reward, i_episode))

        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), torch_save + str(i_episode) + '.pth')

    pool.close()
    pool.join()
    log_queue.put("TERMINATE")
    logger.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--mode", type=str, default='proposed', help="실험 모드: baseline1 | baseline2 | baseline3 | proposed")
    parser.add_argument("-track", "--track", type=str, default='M1')
    parser.add_argument("-cpus", "--cpu_cores", type=int, default=4)
    parser.add_argument("-Eps", "--Eps", type=int, default=1000)
    parser.add_argument("-eps", "--eps", type=int, default=3)
    parser.add_argument("-v", "--vizualize", action='store_true', help="FlightGear 시각화 실행 여부")
    args = vars(parser.parse_args())

    runPPO(args)

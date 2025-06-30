# mainBVRGym_MultiCore.py - 최종 수정본

import os
import json
import argparse, time
from jsb_gym.environments import evasive, bvrdog
import numpy as np
from enum import Enum
from torch.utils.tensorboard import SummaryWriter
import torch
import multiprocessing
from jsb_gym.RL.ppo import Memory, PPO
from jsb_gym.environments.config import BVRGym_PPO1, BVRGym_PPO2, BVRGym_PPODog
from numpy.random import seed
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym, aim_dog_BVRGym, f16_dog_BVRGym

def init_pool():
    seed()

class Maneuvers(Enum):
    Evasive = 0
    Crank = 1

def runPPO(args):
    # --- 시나리오 및 환경 설정 ---
    if args['track'] == 'M1':
        from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
        env_config = BVRGym_PPO1
        env = evasive.Evasive(env_config, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save_path = 'jsb_gym/logs/RL/M1.pth'
        state_scale = 1
    elif args['track'] == 'M2':
        from jsb_gym.RL.config.ppo_evs_PPO2 import conf_ppo
        env_config = BVRGym_PPO2
        env = evasive.Evasive(env_config, args, aim_evs_BVRGym, f16_evs_BVRGym)
        torch_save_path = 'jsb_gym/logs/RL/M2.pth'
        state_scale = 2
    elif args['track'] == 'Dog':
        from jsb_gym.RL.config.ppo_evs_PPO_BVRDog import conf_ppo
        env_config = BVRGym_PPODog
        env = bvrdog.BVRDog(env_config, args, aim_dog_BVRGym, f16_dog_BVRGym)
        torch_save_path = 'jsb_gym/logs/RL/Dog.pth'
        state_scale = 1
    # ... 다른 시나리오 추가 가능

    # --- 수정 1: 커맨드 라인 인자를 conf 객체에 반영 ---
    method = args.get('method', 'baseline')
    conf_ppo.setdefault('general', {})
    conf_ppo['general']['method'] = method
    
    # PPO 알고리즘 하이퍼파라미터 설정
    conf_ppo['entropy_weight'] = args['entropy_w']
    conf_ppo['shaping_w'] = args['shaping_w'] # shaping 가중치도 conf를 통해 전달

    # 환경 보상 가중치 설정 (evasive.py에서 사용)
    if hasattr(env_config, 'reward_weights'):
        env_config.reward_weights['tactical'] = args['tactical_w']
        env_config.reward_weights['shaping'] = args['shaping_w']
        env_config.reward_weights['entropy'] = args['entropy_w']

    # --- 수정 2: 텐서보드 Writer 동적 생성 ---
    run_name = args.get('run_name')
    if not run_name:
        run_name = f"{args['track']}_{method}_tw{args['tactical_w']}_sw{args['shaping_w']}_ew{args['entropy_w']}_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")

    # --- PPO 및 멀티프로세싱 풀 초기화 ---
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    memory = Memory()
    ppo = PPO(state_dim * state_scale, action_dim, conf_ppo, use_gpu=args.get('gpu', False))
    pool = multiprocessing.Pool(processes=int(args['cpu_cores']), initializer=init_pool)

    print(f"--- 학습 시작: {run_name} ---")

    for i_episode in range(1, args['Eps'] + 1):
        ppo_policy = ppo.policy.state_dict()
        ppo_policy_old = ppo.policy_old.state_dict()
        input_data = [(args, ppo_policy, ppo_policy_old, conf_ppo, state_scale) for _ in range(args['cpu_cores'])]
        
        running_rewards = []
        all_infos = []

        # 병렬 학습 실행
        results = pool.map(train, input_data)
        
        # 결과 집계
        for tmp in results:
            memory.actions.extend(tmp[0])
            memory.states.extend(tmp[1])
            memory.logprobs.extend(tmp[2])
            memory.rewards.extend(tmp[3])
            memory.is_terminals.extend(tmp[4])
            running_rewards.append(tmp[5])
            all_infos.append(tmp[6]) # 👈 수정 3: info 딕셔너리 수집

        # 정책 업데이트
        ppo.update(memory, to_tensor=True) 
        memory.clear_memory()
        torch.cuda.empty_cache()

        # --- 수정 4: 전체 로깅 로직 변경 ---
        # 1. 기본 학습 지표 그룹
        writer.add_scalar("Metrics/Running_Reward", np.mean(running_rewards), i_episode)
        writer.add_scalar("Agent/Policy_Entropy", ppo.get_entropy(), i_episode)

        # 2. 상세 평가지표 그룹 (새로운 지표들)
        valid_infos = [info for info in all_infos if 'survival' in info]
        if valid_infos:
            avg_survival = np.mean([info['survival'] for info in valid_infos])
            
            # 회피 성공한 경우만 필터링
            successful_evasions = [info for info in valid_infos if info['survival'] > 0 and not np.isnan(info['evasion_time'])]
            
            avg_min_cpa = np.mean([info['min_cpa'] for info in successful_evasions]) if successful_evasions else 0
            avg_evasion_time = np.mean([info['evasion_time'] for info in successful_evasions]) if successful_evasions else 0
            
            writer.add_scalar("Performance/Survival_Rate", avg_survival, i_episode)
            writer.add_scalar("Performance/Min_CPA", avg_min_cpa, i_episode)
            writer.add_scalar("Performance/Evasion_Time", avg_evasion_time, i_episode)

        # 3. 보상 구성요소 그룹 (환경에 last_reward_components가 있는 경우)
        if hasattr(env, 'last_reward_components'):
            for k, v in env.last_reward_components.items():
                clean_key = k.replace('reward_', '').capitalize()
                writer.add_scalar(f"Rewards/{clean_key}", v, i_episode)
        
            if i_episode % 100 == 0:
                # 'results'에서 궤적 데이터 수집
                trajectories = [tmp[7] for tmp in results]

                # 첫 번째 워커의 궤적만 저장
                if trajectories:
                    trajectory_to_save = trajectories[0] 
                    save_filename = f"trajectories/{run_name}_ep{i_episode}.json"

                    # trajectories 폴더가 없으면 생성
                    os.makedirs("trajectories", exist_ok=True)

                    with open(save_filename, 'w') as f:
                        json.dump(trajectory_to_save, f, indent=4)
                    print(f"Episode {i_episode}: Trajectory saved to {save_filename}")


        # 모델 저장
        if i_episode % 500 == 0:
            torch.save(ppo.policy.state_dict(), f"{torch_save_path}_ep{i_episode}.pth")
            print(f"Episode {i_episode}: Model saved.")

    pool.close()
    pool.join()
    writer.close()
    print(f"--- 학습 종료: {run_name} ---")


def train(packed_args):
    # 인자 언패킹
    args, ppo_policy, ppo_policy_old, conf_ppo, state_scale, = packed_args

    # 각 프로세스에서 환경 다시 생성
    if args['track'] == 'M1':
        from jsb_gym.environments.config import BVRGym_PPO1
        env = evasive.Evasive(BVRGym_PPO1, args, aim_evs_BVRGym, f16_evs_BVRGym)
    elif args['track'] == 'M2':
        from jsb_gym.environments.config import BVRGym_PPO2
        env = evasive.Evasive(BVRGym_PPO2, args, aim_evs_BVRGym, f16_evs_BVRGym)
    elif args['track'] == 'Dog':
        from jsb_gym.environments.config import BVRGym_PPODog
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)
    elif args['track'] == 'DogR':
        from jsb_gym.environments.config import BVRGym_PPODog
        env = bvrdog.BVRDog(BVRGym_PPODog, args, aim_dog_BVRGym, f16_dog_BVRGym)

    maneuver = Maneuvers.Evasive
    memory = Memory()
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]
    ppo = PPO(state_dim * state_scale, action_dim, conf_ppo, use_gpu=False)

    ppo.policy.load_state_dict(ppo_policy)
    ppo.policy_old.load_state_dict(ppo_policy_old)
    ppo.policy.eval()
    ppo.policy_old.eval()

    running_reward = 0.0
    final_info = {}

    for _ in range(args['eps']):
        # --- 상태 초기화 ---
        if args['track'] in ['M1', 'M2']:
            state_block = env.reset(True, True)
            state = state_block['aim1'] if args['track'] == 'M1' else np.concatenate((state_block['aim1'][0], state_block['aim2'][0]))
        else:
            state = env.reset()

        # --- 에피소드 실행 ---
        done = False
        while not done:
            act = ppo.select_action(state, memory)
            action = np.zeros(3)
            action[0], action[1] = act[0], act[1]
            if args['track'] in ['M1', 'M2']:
                action[2] = 1 # Evasive 시나리오에서는 스로틀 고정 등

            # env.step() 호출
            next_state_block, reward, done, info = env.step(action, action_type=maneuver.value)
            
            # 다음 상태 결정
            if args['track'] == 'M1':
                state = next_state_block['aim1']
            elif args['track'] == 'M2':
                state = np.concatenate((next_state_block['aim1'], next_state_block['aim2']))
            else:
                state = next_state_block

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

        running_reward += reward
        final_info = info # 마지막 스텝의 info 저장

    running_reward /= args['eps']
    
    # 디바이스 호환성을 위해 numpy로 변환
    actions = [i.cpu().detach().numpy() for i in memory.actions]
    states = [i.cpu().detach().numpy() for i in memory.states]
    logprobs = [i.cpu().detach().numpy() for i in memory.logprobs]

    return [actions, states, logprobs, memory.rewards, memory.is_terminals, running_reward, final_info, env.trajectory_log]

def interactive_menu():
    """사용자로부터 인터랙티브하게 실험 설정을 입력받는 함수"""
    args = {}
    print("==============================================")
    print(" BVRGym 강화학습 시뮬레이터")
    print("==============================================")

    # --- 1. 트랙(시나리오) 선택 ---
    while True:
        print("\n[1] 학습할 시나리오를 선택하세요:")
        print("  1: M1 (단일 미사일 회피)")
        print("  2: M2 (다중 미사일 회피)")
        print("  3: Dog (BVR 교전)")
        track_choice = input(">> 선택 (번호 입력): ")
        if track_choice == '1':
            args['track'] = 'M1'
            break
        elif track_choice == '2':
            args['track'] = 'M2'
            break
        elif track_choice == '3':
            args['track'] = 'Dog'
            break
        else:
            print("🚨 잘못된 입력입니다. 1, 2, 3 중 하나를 입력하세요.")

    # --- 2. 기법(메소드) 선택 ---
    while True:
        print("\n[2] 적용할 기법을 선택하세요:")
        print("  1: baseline (생존 보상만 사용)")
        print("  2: shaping (Shaping 보상 추가)")
        print("  3: entropy (Entropy 보너스 강화)")
        print("  4: proposed (제안하는 통합 보상 프레임워크)")
        method_choice = input(">> 선택 (번호 입력): ")
        if method_choice == '1':
            args['method'] = 'baseline'
            break
        elif method_choice == '2':
            args['method'] = 'shaping'
            break
        elif method_choice == '3':
            args['method'] = 'entropy'
            break
        elif method_choice == '4':
            args['method'] = 'proposed'
            break
        else:
            print("🚨 잘못된 입력입니다. 1, 2, 3, 4 중 하나를 입력하세요.")

    # --- 3. 세부 가중치 설정 ---
    print("\n[3] 세부 가중치를 설정합니다. (기본값을 사용하려면 Enter)")
    
    # 기본값 설정
    args['shaping_w'] = 0.05
    args['entropy_w'] = 0.01
    args['tactical_w'] = 0.3

    if args['method'] in ['shaping', 'proposed']:
        sw_input = input(f">> Shaping 가중치 입력 (기본값: {args['shaping_w']}): ")
        if sw_input: args['shaping_w'] = float(sw_input)
    
    if args['method'] in ['entropy', 'proposed']:
        ew_input = input(f">> Entropy 가중치 입력 (기본값: {args['entropy_w']}): ")
        if ew_input: args['entropy_w'] = float(ew_input)

    if args['method'] == 'proposed':
        tw_input = input(f">> Tactical 가중치 입력 (기본값: {args['tactical_w']}): ")
        if tw_input: args['tactical_w'] = float(tw_input)

    # --- 4. 기타 설정 ---
    print("\n[4] 기타 학습 설정을 입력합니다. (기본값을 사용하려면 Enter)")
    
    cpu_cores_input = input(">> 사용할 CPU 코어 수 (기본값: 12): ")
    args['cpu_cores'] = int(cpu_cores_input) if cpu_cores_input else 12
    
    eps_input = input(">> 병렬 에피소드 수 (기본값: 3): ")
    args['eps'] = int(eps_input) if eps_input else 3
    
    Eps_input = input(">> 총 학습 에피소드 수 (기본값: 10000): ")
    args['Eps'] = int(Eps_input) if Eps_input else 10000

        # --- 5. GPU 사용 여부 선택 ---
    while True:
        print("\n[5] GPU(CUDA)를 사용하여 학습을 가속하시겠습니까?")
        print("   (NVIDIA GPU 및 CUDA 환경 설정이 필요합니다)")
        gpu_choice = input(">> 선택 (y/n): ").lower()
        if gpu_choice == 'y':
            args['gpu'] = True
            break
        elif gpu_choice == 'n':
            args['gpu'] = False
            break
        else:
            print("🚨 잘못된 입력입니다. 'y' 또는 'n'을 입력하세요.")

    # argparse와 형식을 맞추기 위한 나머지 기본값들
    args['vizualize'] = False
    args['run_name'] = ""

    print("\n--- 설정 완료 ---")
    for key, value in args.items():
        print(f"  {key}: {value}")
    print("--------------------")
    
    return args


if __name__ == '__main__':
    # 인터랙티브 메뉴를 통해 실험 설정값을 받아옴
    args = interactive_menu()
    
    # 설정된 값으로 PPO 학습 실행
    runPPO(args)
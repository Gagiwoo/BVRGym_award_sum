# evaluate.py: PPO 학습 모델을 다양한 환경에서 평가 + 행동 다양성 로그 저장

import torch
import numpy as np
import os, csv, argparse
from jsb_gym.RL.ppo import PPO
from jsb_gym.environments.config import BVRGym_PPO1
from jsb_gym.TAU.config import aim_evs_BVRGym, f16_evs_BVRGym

# 환경 선택
import jsb_gym.environments.evasive_b1 as ev_b1
import jsb_gym.environments.evasive_b2 as ev_b2
import jsb_gym.environments.evasive_b3 as ev_b3
import jsb_gym.environments.evasive_proposed as ev_proposed
ENV_MAP = {
    'baseline1': ev_b1.Evasive,
    'baseline2': ev_b2.Evasive,
    'baseline3': ev_b3.Evasive,
    'proposed': ev_proposed.Evasive
}

def evaluate(args):
    SAVE_DIR = os.path.join("eval_logs", args.mode)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 환경 로딩
    env_class = ENV_MAP[args.mode]
    env_args = {"vizualize": False}
    env = env_class(BVRGym_PPO1, env_args, aim_evs_BVRGym, f16_evs_BVRGym)
    state_dim = env.observation_space
    action_dim = env.action_space.shape[1]

    # PPO 로드
    from jsb_gym.RL.config.ppo_evs_PPO1 import conf_ppo
    ppo = PPO(state_dim, action_dim, conf_ppo, use_gpu=torch.cuda.is_available())
    ppo.policy_old.load_state_dict(torch.load(args.model_path, map_location=ppo.device))
    ppo.policy_old.eval()

    episode_rewards = []
    episode_diversity = []
    episode_wins = 0

    for ep in range(1, args.episodes + 1):
        state_block = env.reset(rand_state_f16=True, rand_state_aim=True)
        state = state_block['aim1']
        done = False
        action = np.zeros(3)

        trajectory = []
        actions = []
        total_reward = 0

        while not done:
            act = ppo.select_action(state, memory=None, gready=True)
            action[0], action[1] = act[0], act[1]
            action[2] = 1.0
            state_block, reward, done, _ = env.step(action, action_type=0)
            state = state_block['aim1']

            total_reward += reward
            actions.append(action[:2].copy())

            trajectory.append([
                env.f16.get_lat_gc_deg(), env.f16.get_long_gc_deg(), env.f16.get_altitude(),
                reward, *action
            ])

        # 다양성 계산
        diffs = [np.linalg.norm(np.array(actions[i]) - np.array(actions[i - 1]))
                 for i in range(1, len(actions))]
        diversity = np.mean(diffs) if diffs else 0

        # 궤적 저장
        save_path = os.path.join(SAVE_DIR, f"episode_{ep:03d}.csv")
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["lat", "lon", "alt", "reward", "hdg", "alt_cmd", "thr"])
            writer.writerows(trajectory)

        episode_rewards.append(total_reward)
        episode_diversity.append(diversity)
        episode_wins += int(env.f16_alive)
        print(f"[✓] Saved episode {ep:03d} | Reward: {total_reward:.2f} | Diversity: {diversity:.3f}")

    # 요약 통계 저장
    with open(os.path.join(SAVE_DIR, "summary.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "diversity", "win"])
        for i in range(args.episodes):
            writer.writerow([i+1, episode_rewards[i], episode_diversity[i], int(i < episode_wins)])

    print("\n📊 평가 완료!")
    print(f"- 평균 보상: {np.mean(episode_rewards):.2f}")
    print(f"- 평균 다양성: {np.mean(episode_diversity):.3f}")
    print(f"- 생존률 (승률): {episode_wins / args.episodes:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="proposed",
                        help="평가할 모델 모드: baseline1 | baseline2 | baseline3 | proposed")
    parser.add_argument("--model_path", type=str, default="logs/RL/M1_proposed.pth",
                        help="학습된 모델 경로")
    parser.add_argument("--episodes", type=int, default=10,
                        help="평가 에피소드 수")
    args = parser.parse_args()
    evaluate(args)
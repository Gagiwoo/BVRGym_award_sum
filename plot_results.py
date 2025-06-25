# plot_results.py: 다양한 모델의 평가 결과를 비교 시각화

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font="Malgun Gothic", font_scale=1.1)

# 📂 평가 결과 디렉토리
BASE_DIR = "eval_logs"
MODES = ["baseline1", "baseline2", "baseline3", "proposed"]

# 📊 지표별 저장용 리스트
data = {
    "mode": [],
    "reward": [],
    "diversity": [],
    "win": []
}

# 📥 각 summary.csv 로딩
for mode in MODES:
    path = os.path.join(BASE_DIR, mode, "summary.csv")
    if not os.path.exists(path):
        print(f"[경고] {path} 없음")
        continue
    df = pd.read_csv(path)
    data["mode"].extend([mode] * len(df))
    data["reward"].extend(df["reward"].tolist())
    data["diversity"].extend(df["diversity"].tolist())
    data["win"].extend(df["win"].tolist())

# 📐 DataFrame으로 정리
df = pd.DataFrame(data)

# 🎨 보상 박스플롯
plt.figure(figsize=(10, 5))
sns.boxplot(x="mode", y="reward", data=df, palette="Set2")
plt.title("모델별 에피소드 총 보상 분포")
plt.savefig("reward_comparison.png")
plt.close()

# 🎨 다양성 박스플롯
plt.figure(figsize=(10, 5))
sns.boxplot(x="mode", y="diversity", data=df, palette="Set1")
plt.title("모델별 행동 다양성 분포")
plt.savefig("diversity_comparison.png")
plt.close()

# 🎯 생존률 막대그래프
plt.figure(figsize=(8, 5))
win_df = df.groupby("mode")["win"].mean().reset_index()
sns.barplot(x="mode", y="win", data=win_df, palette="Blues_d")
plt.title("모델별 평균 생존률")
plt.ylim(0, 1.0)
plt.ylabel("생존률")
plt.savefig("winrate_comparison.png")
plt.close()

print("\n[✓] 모든 비교 그래프 저장 완료!")
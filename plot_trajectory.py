# plot_trajectory.py: evaluate.py에서 저장한 로그(.csv)를 시각화하는 스크립트

import os
import matplotlib.pyplot as plt
import pandas as pd

# 로그 저장 폴더 경로
LOG_DIR = "eval_logs"

# 평가된 로그 파일 불러오기 (최대 5개까지)
csv_files = sorted([f for f in os.listdir(LOG_DIR) if f.endswith('.csv')])[:5]

# 시각화 시작
for file in csv_files:
    path = os.path.join(LOG_DIR, file)
    df = pd.read_csv(path)

    # 기본 Trajectory Plot (2D)
    plt.plot(df['lon'], df['lat'], label=file.replace('.csv',''))

plt.title("F-16 Trajectories (Lat/Lon)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 고도 그래프 별도 시각화
for file in csv_files:
    path = os.path.join(LOG_DIR, file)
    df = pd.read_csv(path)
    plt.plot(df['alt'], label=file.replace('.csv',''))

plt.title("Altitude over Time")
plt.xlabel("Timestep")
plt.ylabel("Altitude (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

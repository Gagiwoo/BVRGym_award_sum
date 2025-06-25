import pygame
import numpy as np
import jsbsim
from jsb_gym.environmets import evasive
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# JSBSim 환경 로드
fdm = jsbsim.FGFDMExec("./")
fdm.load_model("f16")
fdm.run_ic()

# 기록된 위치 데이터 저장
f16_positions = []
missile_positions = []

# 시뮬레이션 실행 중 위치 저장
for step in range(10000):
    f16_x = fdm.get_property_value('position/x')
    f16_y = fdm.get_property_value('position/y')
    f16_z = fdm.get_property_value('position/z')

    missile_x = fdm.get_property_value('position/x') + 5
    missile_y = fdm.get_property_value('position/y') + 5
    missile_z = fdm.get_property_value('position/z') + 5

    f16_positions.append([f16_x, f16_y, f16_z])
    missile_positions.append([missile_x, missile_y, missile_z])

# 🎯 Matplotlib 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 🎯 데이터 시각화
f16_positions = np.array(f16_positions)
missile_positions = np.array(missile_positions)

ax.plot(f16_positions[:, 0], f16_positions[:, 1], f16_positions[:, 2], 'r-', label="16")
ax.plot(missile_positions[:, 0], missile_positions[:, 1], missile_positions[:, 2], 'b-', label="MISS")
plt.gca().autoscale()

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.legend()
plt.show()
@echo off
REM 여러 w 값으로 자동 학습 실험 (Windows 배치파일)
setlocal

set TRACK=M1
set EPS=1000
set CPU=4

for %%W in (0.01 0.03 0.05 0.1 0.2) do (
    echo [★] Start w=%%W
    python mainBVRGym_MultiCore.py --mode proposed --track %TRACK%_w%%W --cpu_cores %CPU% --Eps %EPS% --eps 3 --shaping_w %%W
)

echo [✓] 모든 실험 종료!

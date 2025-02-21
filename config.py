import numpy as np

# ==========================
# **하이퍼파라미터 설정**
# ==========================
# 드론의 목표 위치 및 환경 설정
TARGET_POSITION = np.array([3, 2, -2])  # 목표 위치
MAX_STEPS = 200  # 한 에피소드당 최대 스텝 수
SAFE_BOUND = 50  # 안전 경계값 (이 값을 초과하면 종료)
OBSERVATION_SPACE_LOW = -100  # 관측 공간 최소값
OBSERVATION_SPACE_HIGH = 100  # 관측 공간 최대값
ACTION_SPACE_LOW = np.array([-3, -3, -1, -30])  # 행동 최소값 (vx, vy, vz, yaw_rate)
ACTION_SPACE_HIGH = np.array([3, 3, 1, 30])  # 행동 최대값

# PPO 학습 관련 설정
PPO_N_STEPS = 64  # 학습 시 하나의 배치에서 사용하는 스텝 수
PPO_BATCH_SIZE = 16  # 미니배치 크기
PPO_N_EPOCHS = 5  # 학습 반복 횟수
PPO_POLICY = "MlpPolicy"  # 사용되는 정책 네트워크

# 보상 관련 설정
REWARD_GOAL = 100  # 목표 도달 보상
REWARD_STEP = -0.1  # 기본 스텝당 보상
REWARD_DISTANCE_GAIN = 10  # 목표 접근 보상
REWARD_DISTANCE_LOSS = -10  # 목표에서 멀어질 때 패널티
REWARD_OUT_OF_BOUNDS = -10  # 안전 경계를 벗어났을 때 패널티
REWARD_MAX_STEP_EXCEED = -50  # 최대 스텝 초과 시 패널티
REWARD_COLLISION = -50

# 학습 모델 저장 경로
# MODEL_PATH = "ppo_airsim_drone_policy3.zip" # 이거는 state가 3일때임(x,y,z 위치정보)
MODEL_PATH = "ppo_airsim_lidar_sensor.zip"
TOTAL_TIMESTEPS = 128  # 학습할 총 스텝 수

TEST_EPISODE_STEPS = 20  # 테스트 실행 시 반복할 스텝 수

# 학습 중단 키 
STOP_KEY = "q"

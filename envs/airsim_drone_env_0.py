import airsim
import numpy as np
import gym
import open3d as o3d
from gym import spaces
import config
import os
import re

# ==========================
# **드론 환경 클래스 정의**
# ==========================
class AirSimDroneEnv(gym.Env):
    def __init__(self):
        super(AirSimDroneEnv, self).__init__()

        # 1. AirSim 클라이언트 초기화
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 2. LiDAR 센서 설정
        self.lidar_name = "LidarSensor1"
        
        # 3. 관측 공간 설정
        self.observation_space = spaces.Dict({
            '''
            "drone_position": spaces.Box(
                low=np.array([config.OBSERVATION_SPACE_LOW] * 3, dtype=np.float32),
                high=np.array([config.OBSERVATION_SPACE_HIGH] * 3, dtype=np.float32),
                dtype=np.float32
            ),
            '''
            "lidar_points": spaces.Box(
                low=config.OBSERVATION_SPACE_LOW,
                high=config.OBSERVATION_SPACE_HIGH,
                shape=(80, 3),
                dtype=np.float32
            )
        })

        # 4. 행동 공간 설정
        self.action_space = spaces.Box(
            low=config.ACTION_SPACE_LOW, 
            high=config.ACTION_SPACE_HIGH, 
            shape=(4,), 
            dtype=np.float32
        )

        # 5. 환경 관련 변수 초기화
        self.target_pos = config.TARGET_POSITION  # 목표 위치
        self.max_steps = config.MAX_STEPS  # 에피소드 당 최대 스텝
        self.safe_bound = config.SAFE_BOUND  # 너무 멀어지면 에피소드 종료

    def reset(self):
        """환경 초기화(에피소드마다 호출됨)"""

        # 1. 드론 초기화 및 API 설정
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # 2. 드론 초기 위치로 이동 및 이륙
        self.client.moveToPositionAsync(0, 0, -1, 1).join()
        self.client.takeoffAsync().join()

        # 3. 환경 변수 초기화
        self.lidar_data_log = []  # LiDAR 데이터 초기화 (이전 데이타 참고 안하기?)
        self.step_in_episode = 0  # 에피소드 내 몇번째 스텝인지

        # 4. 현재 상태 확인 및 반환
        state = self._get_state()
        print(f"[INFO] 환경 초기화 완료 | 초기 상태: {state}")
        return state

    def close(self):
        """환경 종료 시 클라이언트 연결 해제"""

        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()
        print("[INFO] 환경 종료 및 리소스 정리 완료.")

    def step(self, action):
        """드론의 행동 수행 및 다음 상태 반환"""

        # 1. 이번 스텝 action 확인
        print(f"[DEBUG] Action Taken: {action}")
        
        # 2. 이번 스텝 state 가져오기
        state = self._get_state()

        # 3. 드론 이동 명령 실행
        vx, vy, vz, yaw_rate = map(float, action)
        self.client.moveByVelocityAsync(vx, vy, vz, duration=0.5).join() # 1초 너무 길수도 있어서 줄여. 학습속도 향상.
        
        # 4. 새로운 state 가져오기 
        new_state = self._get_state()

        # 5. 이전 state와 새로운 state의 거리 계산산
        prev_distance = np.linalg.norm(state["drone_position"] - self.target_pos)
        current_distance = np.linalg.norm(new_state["drone_position"] - self.target_pos)
        
        # 6. reward 계산 및 종료 여부 
        reward, done = self.calculate_reward(prev_distance, current_distance, new_state)

        # 7. 새로운 state, reward, 종료 여부, 추가 정보 반환
        return new_state, reward, done, {}

    def calculate_reward(self, prev_distance, current_distance, new_state):
        '''
        보상 계산 로직
        1) 충돌 감지 ->           패널티 + 종료
        2) 최대 스텝 초과 ->      패널티 + 종료
        3) 목표 도달 감지 ->      보상   + 종료
        4) 안전 경계 이탈 감지 -> 패널티 + 종료
        5) 기본 스텝 ->           패널티
        6) 목표와의 거리 변화 -> 가까워지면 보상, 멀어지면 패널티 
        '''
        # 1. 충돌 감지 
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            print(f"[DEBUG] 충돌 발생!")
            self.step_in_episode = 0
            return config.REWARD_COLLISION, True  
        
        # 2. 최대 스텝 초과 감지 
        if self.step_in_episode >= self.max_steps:
            print("[DEBUG] 최대 스텝 초과 - 에피소드 종료")
            self.step_in_episode = 0
            return config.REWARD_MAX_STEP_EXCEED, True  
        
        # 3. 목표 도달 감지 
        if current_distance < 1:
            print("[DEBUG] 목표 도달! 보상 지급.")
            self.step_in_episode = 0
            return config.REWARD_GOAL, True  

        # 4. 안전 경계 이탈 감지 
        center = np.array([0.0, 0.0, -2.0])  # 구(원형) 경계의 중심점
        drone_position = new_state["drone_position"]  
        distance_from_center = np.linalg.norm(drone_position - center)
        if distance_from_center > self.safe_bound:
            print("[DEBUG] 구(원형) 경계 이탈!")
            self.step_in_episode = 0
            return config.REWARD_OUT_OF_BOUNDS, True

        # 5. 기본 스텝 패널티 
        reward = config.REWARD_STEP

        # 6. 목표와의 거리 변화 반영
        distance_change = prev_distance - current_distance
        if distance_change > 0:
            reward += config.REWARD_DISTANCE_GAIN * (distance_change ** 1.5)  # 거리 감소 시 보상을 1.5제곱(2는 너무 보상이 급격하게 변동-> 학습 불안정)
            print(f"[DEBUG] 목표 접근 | 보상 증가: {reward:.2f}")
        else:
            reward += config.REWARD_DISTANCE_LOSS * (abs(distance_change) ** 1.5)  # 거리 증가 시 패널티 
            print(f"[DEBUG] 목표에서 멀어짐 | 패널티 적용: {reward:.2f}")

        # 7. 스텝 카운트 증가 후 보상 반환환
        self.step_in_episode += 1
        return reward, False  

    def _get_state(self):
        """현재 드론 위치(x,y,z)와 LiDAR 데이터를 반환"""

        # 1. 드론의 현재 위치 가져오기 
        multirotor_state = self.client.getMultirotorState()
        pos = multirotor_state.kinematics_estimated.position
        drone_position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)
        
        # 2. LiDAR 데이터 가져오기
        lidar_data = self.client.getLidarData(lidar_name=self.lidar_name)
        print(f"[DEBUG] LiDAR 데이터 개수: {len(lidar_data.point_cloud)}")

        # 3. LiDAR 포인트 데이터 변환
        if len(lidar_data.point_cloud) == 0:
            # LiDAR 데이터가 없으면 0으로 채움
            lidar_points = np.zeros((80, 3), dtype=np.float32)  
        else:
            # LiDAR 포인트 개수 계산 후 (x,y,z) 형태로 변환
            num_points = len(lidar_data.point_cloud) // 3  
            lidar_points = np.array(lidar_data.point_cloud[:num_points * 3], dtype=np.float32).reshape(-1, 3)

        # 4. LiDAR 포인트 개수 조정 (np.pad() 사용)
        lidar_points = np.pad(lidar_points, ((0, max(0, 80 - lidar_points.shape[0])), (0, 0)), mode='constant')[:80] 
        print(f"[DEBUG] 최종 LiDAR 데이터 개수: {lidar_points.shape[0]}")  
        # lidar_points.shape -> (80,3)
        # lidar_points.shape[0] -> 80

        # 5. state 반환
        state = {
            "drone_position": drone_position,  # (3,) || 1차원 배열이며, 크기는 (3,)
            "lidar_points": lidar_points       # (80,3) || 2차원 배열이며, 크기는 (80,3)
        }
        return state
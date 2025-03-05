import airsim
import numpy as np
import gym
import open3d as o3d
from gym import spaces
import config
import os
import re
import cv2

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

        # 2. LiDAR 센서 설정 (카메라는 별도의 설정 없이 사용 가능)
        self.lidar_name = "LidarSensor1"

        # 3. 관측 공간 설정 
        self.observation_space = spaces.Dict({
            "lidar_points": spaces.Box(
                low=config.OBSERVATION_SPACE_LOW,
                high=config.OBSERVATION_SPACE_HIGH,
                shape=(80, 3),
                dtype=np.float32
            ),
            "camera_image": spaces.Box(
                # 64x64 크기의 RGB 이미지(3채널)를 저장
                low=0, high=255, shape=(480, 640, 3), dtype=np.uint8 
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

        # 6. 드론의 위치를 state가 아닌 전역변수로? (임시임)
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position

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
        """드론의 행동 수행하고, 새로운 state, reward, 종료 여부(done)를 반환"""

        # 1. 이번 스텝 action 확인
        print(f"[DEBUG] Action Taken: {action}")

        # 2-1. 드론 위치 가져오기 (이동 전, state 아니라 별도로 임시 설정)
        prev_state = self.client.getMultirotorState()
        prev_pos = prev_state.kinematics_estimated.position
        prev_drone_position = np.array([prev_pos.x_val, prev_pos.y_val, prev_pos.z_val], dtype=np.float32)

        # 2-2. 목표와의 거리 계산 (이동 전)
        prev_distance = np.linalg.norm(prev_drone_position - self.target_pos)

        # 3. 드론 이동 명령 실행
        vx, vy, vz, yaw_rate = map(float, action)
        self.client.moveByVelocityAsync(vx, vy, vz, duration=0.5).join()

        # 4. 새로운 state 가져오기
        new_state = self._get_state()

        # 5-1. 새로운 드론 위치 가져오기 (API 사용, state 아니라 별도로 임시 설정)
        multirotor_state = self.client.getMultirotorState()
        pos = multirotor_state.kinematics_estimated.position
        new_drone_position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

        # 5-2. 새로운 목표 거리 계산
        current_distance = np.linalg.norm(new_drone_position - self.target_pos)

        # 6. reward 계산 및 종료 여부
        reward, done = self.calculate_reward(prev_distance, current_distance)

        return new_state, reward, done, {}


    def calculate_reward(self, prev_distance, current_distance):
        '''
        보상 계산 로직
        0) 지금은 state에 드론의 위치가 없으므로 전역변수로 가져와야 함함
        1) 충돌 감지 ->           패널티 + 종료
        2) 최대 스텝 초과 ->      패널티 + 종료
        3) 목표 도달 감지 ->      보상   + 종료
        4) 안전 경계 이탈 감지 -> 패널티 + 종료
        5) 기본 스텝 ->           패널티
        6) 목표와의 거리 변화 -> 가까워지면 보상, 멀어지면 패널티 
        '''
        # 0. 드론의 위치 가져와 
        multirotor_state = self.client.getMultirotorState()
        pos = multirotor_state.kinematics_estimated.position
        drone_position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

        # 1. 충돌 감지 
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            print(f"[DEBUG] 충돌 발생")
            self.step_in_episode = 0
            return config.REWARD_COLLISION, True  
        
        # 2. 최대 스텝 초과 감지 
        if self.step_in_episode >= self.max_steps:
            print("[DEBUG] 최대 스텝 초과")
            self.step_in_episode = 0
            return config.REWARD_MAX_STEP_EXCEED, True  
        
        # 3. 목표 도달 감지 
        if current_distance < 1:
            print("[DEBUG] 목표 도달! 보상 지급.")
            self.step_in_episode = 0
            return config.REWARD_GOAL, True  

        # 4. 안전 경계 이탈 감지 
        center = np.array([0.0, 0.0, -2.0])  # 구(원형) 경계의 중심점
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

        # 7. 스텝 카운트 증가 후 보상 반환
        self.step_in_episode += 1
        return reward, False  
    
    def _get_state(self):
        """현재 LiDAR 데이터, 카메라 이미지 반환"""

        # 1-1. LiDAR 데이터 가져오기
        lidar_data = self.client.getLidarData(lidar_name=self.lidar_name)
        print(f"[DEBUG] LiDAR 데이터 개수: {len(lidar_data.point_cloud)}")

        # 1-2. LiDAR 포인트 데이터 변환
        if len(lidar_data.point_cloud) == 0:
            lidar_points = np.zeros((80, 3), dtype=np.float32)  
        else:
            num_points = len(lidar_data.point_cloud) // 3  
            lidar_points = np.array(lidar_data.point_cloud[:num_points * 3], dtype=np.float32).reshape(-1, 3)

        # 1-3. LiDAR 포인트 개수 조정(80개로 고정)
        lidar_points = np.pad(lidar_points, ((0, max(0, 80 - lidar_points.shape[0])), (0, 0)), mode='constant')[:80] 
        print(f"[DEBUG] 최종 LiDAR 데이터 개수: {lidar_points.shape[0]}")

        # 2-1. 카메라 이미지 가져오기 
        responses = self.client.simGetImages([
            airsim.ImageRequest("FrontCamera", airsim.ImageType.Scene, False, False)
        ])

        # 2-2 기본 이미지 설정 (오류 방지)
        img = np.zeros((480, 640, 3), dtype=np.uint8)  # 기본 검은색 이미지

        # 2-3 원본 이미지 크기 출력 및 저장
        if responses and len(responses) > 0 and responses[0].image_data_uint8:
            print(f"[DEBUG] 원본 카메라 이미지 크기: {responses[0].height} x {responses[0].width}")  
            # ✅ 이미지 데이터를 NumPy 배열로 변환 후 디코딩
            img_data = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            temp_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            # ✅ 디코딩이 성공하면 img를 업데이트
            if temp_img is not None:
                img = temp_img
            else:
                print("[WARNING] 카메라 이미지 디코딩 실패, 기본 이미지 사용")
        else:
            print("[ERROR] 카메라 응답 없음 또는 데이터 없음! 기본 이미지 사용")

        # 4 드론 위치 제거 후 state 반환
        state = {
            "lidar_points": lidar_points,  
            "camera_image": img  
        }

        return state
        

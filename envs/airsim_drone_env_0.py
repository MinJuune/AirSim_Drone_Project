import airsim
import numpy as np
import gym
import open3d as o3d
from gym import spaces
import config

# ==========================
# **드론 환경 클래스 정의**
# ==========================
class AirSimDroneEnv(gym.Env):
    def __init__(self):
        super(AirSimDroneEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # LiDAR 센서 활성화
        self.lidar_name = "LidarSensor"

        # Open3D 시각화 설정
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        self.observation_space = spaces.Box(
            low=config.OBSERVATION_SPACE_LOW, 
            high=config.OBSERVATION_SPACE_HIGH, 
            shape=(100, 3),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=config.ACTION_SPACE_LOW, 
            high=config.ACTION_SPACE_HIGH, 
            shape=(4,), 
            dtype=np.float32
        )

        self.target_pos = config.TARGET_POSITION
        self.max_steps = config.MAX_STEPS
        self.step_count = 0
        self.safe_bound = config.SAFE_BOUND

    def reset(self):
        """환경 초기화"""
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.moveToPositionAsync(0, 0, -2, 1).join()
        self.client.takeoffAsync().join()

        self.step_count = 0
        self.step_in_episode = 0
        state = self._get_state()
        print(f"[INFO] 환경 초기화 완료 | 초기 상태: {state}")
        return state

    def close(self):
        """환경 종료 시 클라이언트 연결 해제"""
        self.vis.destroy_window()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()
        print("[INFO] 환경 종료 및 리소스 정리 완료.")

    def step(self, action):
        """드론의 행동 수행 및 다음 상태 반환"""
        print(f"[DEBUG] Action Taken: {action}")
        state = self._get_state()
        vx, vy, vz, yaw_rate = map(float, action)
        self.client.moveByVelocityAsync(vx, vy, vz, duration=1).join()
        new_state = self._get_state()

        # 충돌 감지
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            print(f"[DEBUG] 충돌 발생!")
            return new_state, config.REWARD_COLLISION, True, {}  # 충돌 시 패널티 적용 후 종료

        prev_distance = np.linalg.norm(state[:3] - self.target_pos)
        current_distance = np.linalg.norm(new_state[:3] - self.target_pos)
        reward, done = self.calculate_reward(prev_distance, current_distance, new_state)
        self.step_count += 1

        return new_state, reward, done, {}

    def calculate_reward(self, prev_distance, current_distance, new_state):
        """보상 계산 로직"""

        if self.step_in_episode >= 5:
            print("[DEBUG] 에피소드 내 최대 스텝 초과 - 에피소드 종료")
            self.step_in_episode = 0
            return config.REWARD_MAX_STEP_EXCEED, True

        # 최대 스텝을 초과하면 패널티 적용 후 종료
        if self.step_count >= self.max_steps:
            print("[DEBUG] 최대 스텝 초과 - 에피소드 종료")
            self.step_in_episode = 0
            return config.REWARD_MAX_STEP_EXCEED, True  # 최대 스텝 초과하면 즉시 종료

        # 목표 도달 시 종료
        if current_distance < 1:
            print("[DEBUG] 목표 도달! 보상 지급.")
            self.step_in_episode = 0
            return config.REWARD_GOAL, True  # 목표 도달하면 즉시 종료

        # 경계를 벗어나면 즉시 종료
        if np.any(np.abs(new_state[:3]) > self.safe_bound):
            print(f"[DEBUG] 안전 경계 초과! 현재 위치: {new_state}")
            self.step_in_episode = 0
            return config.REWARD_OUT_OF_BOUNDS, True  # 안전 경계를 벗어나면 즉시 종료

        # 기본 보상 설정
        reward = config.REWARD_STEP

        # 목표 접근 여부에 따른 보상 계산
        distance_change = prev_distance - current_distance
        if distance_change > 0:
            reward += config.REWARD_DISTANCE_GAIN * (distance_change ** 2)  # 거리 감소 시 보상을 제곱하여 가중치 부여
            print(f"[DEBUG] 목표 접근 | 보상 증가: {reward:.2f}")
        else:
            reward += config.REWARD_DISTANCE_LOSS * (abs(distance_change) ** 2)  # 거리 증가 시 패널티 강화
            print(f"[DEBUG] 목표에서 멀어짐 | 패널티 적용: {reward:.2f}")

        self.step_in_episode += 1
        return reward, False  # 일반적인 경우 계속 진행

    def _get_state(self):
        """현재 드론 위치와 LiDAR 데이터를 반환"""
        multirotor_state = self.client.getMultirotorState()
        pos = multirotor_state.kinematics_estimated.position
        drone_position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

        # LiDAR 데이터 가져오기
        lidar_data = self.client.getLidarData(lidar_name=self.lidar_name)
        if len(lidar_data.point_cloud) < 3:
            lidar_points = np.zeros((100, 3), dtype=np.float32)  # 데이터 없을 때
        else:
            lidar_points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

        # Open3D를 활용한 LiDAR 데이터 실시간 시각화
        self.pcd.points = o3d.utility.Vector3dVector(lidar_points)
        self.pcd.paint_uniform_color([0, 1, 0])  # 초록색
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

        # 상태: 드론 위치 + LiDAR 데이터
        return np.concatenate((drone_position.reshape(1, 3), lidar_points[:99]), axis=0)
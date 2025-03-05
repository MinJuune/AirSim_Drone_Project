import os
import json
import numpy as np
import airsim  
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, save_path="lidar_data", verbose=1):
        """콜백 클래스 초기화"""

        super(CustomCallback, self).__init__(verbose)
        
        # 1. LiDAR 데이터 저장 폴더 설정 
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)  

        # 2. AirSim 클라이언트 연결결
        self.client = airsim.MultirotorClient()  
        self.client.confirmConnection()

        # 3. 기존 데이터 확인하여 최신 timestep 찾기
        self.start_timestep = self._get_latest_timestep() + 1
        print(f"저장된 LiDAR 데이터 감지됨. 다음 timestep부터 저장 시작: {self.start_timestep}")

    def _get_latest_timestep(self):
        """현재 저장된 파일들 중 가장 큰 timestep을 찾아 반환"""
        
        # 1. 저장된 JSON 파일 목록 가져오기 
        files = os.listdir(self.save_path)
        timestep_numbers = []

        # 2. 정규식을 사용하여 'timestep_XXXXXX.json'에서 숫자 부분만 추출
        for file in files:
            if file.startswith("timestep_") and file.endswith(".json"):
                try:
                    timestep = int(file.split("_")[1].split(".")[0])
                    timestep_numbers.append(timestep)
                except ValueError:
                    continue

        # 3. 가장 큰 timestep 값 반환 (파일 없으면 -1 반환)
        return max(timestep_numbers) if timestep_numbers else -1  

    def _on_step(self) -> bool:
        """
        PPO 학습 스텝마다 
        현재 학습 정보 출력,
        LiDAR 데이터 저장,
        Unreal Engine 시각화
        """

        # 1. 현재 state 및 reward 가져오기 
        obs = self.locals["new_obs"]  # 현재 state
        reward = self.locals["rewards"][0]  # 현재 reward
        current_timesteps = self.num_timesteps  # 현재 스텝 번호
        lidar_step = self.start_timestep + self.num_timesteps  # 기존 데이터 이후부터 저장하기 위한 변수(라이다) 

        # 2. 드론 위치 & LiDAR 데이터 확인 및 변환
        if "lidar_points" in obs:
            lidar_points = np.array(obs["lidar_points"])  # numpy 배열로 변환
            lidar_points = np.squeeze(lidar_points)  # (1,80,3) → (80,3)으로 변환
            lidar_points_count = lidar_points.shape[0]  # LiDAR 포인트 개수 확인
            
            multirotor_state = self.client.getMultirotorState()
            pos = multirotor_state.kinematics_estimated.position
            drone_position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

            # LiDAR 데이터 저장 (JSON 형식)
            filename = os.path.join(self.save_path, f"timestep_{lidar_step:06d}.json")
            with open(filename, "w") as f:
                json.dump({"drone_position": drone_position, "lidar_points": lidar_points.tolist()}, f)

            print(f"✅ LiDAR 데이터 저장 완료: {filename} (Step: {lidar_step})")

            # Unreal Engine에 LiDAR 포인트 시각화
            self.client.simPlotPoints(
                points=[airsim.Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in lidar_points],  # 변환된 좌표 사용
                color_rgba=[0, 255, 0, 255],  # ✅ 초록색 디버그 포인트
                size=5.0,  # 포인트 크기
                duration=0.2  # 0.2초 후 사라짐
            )
            print(f"📌 {lidar_points_count}개의 LiDAR 포인트가 Unreal에 표시됨!")

        else:
            print("[ERROR] obs 딕셔너리에 'lidar_points' 키가 없음!")

        # 🔥 학습 정보 출력
        print(f"📌 Step {current_timesteps} | Reward: {reward:.2f} | Drone Position: {np.round(drone_position, 2)}")

        return True  # 학습 계속 진행

import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
import time
import airsim
import re

class CustomCallback(BaseCallback):
    def __init__(self, save_path="lidar_data", verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)  # 폴더 생성
        self.last_time = time.time()  # ⏱️ 처음 시간 기록
        self.client = airsim.MultirotorClient()  # ✅ AirSim 클라이언트 연결
        self.client.confirmConnection()

        # 🔥 기존 데이터 확인하여 최신 timestep 찾기
        self.start_timestep = self._get_latest_timestep() + 1
        print(f"📂 저장된 LiDAR 데이터 감지됨. 다음 timestep부터 저장 시작: {self.start_timestep}")

    def _get_latest_timestep(self):
        """✅ 현재 저장된 파일들 중 가장 큰 timestep을 찾아 반환"""
        files = os.listdir(self.save_path)
        timestep_numbers = []

        # 정규식을 사용하여 'timestep_XXXXXX.npy'에서 숫자 부분만 추출
        for file in files:
            match = re.match(r"timestep_(\d+).npy", file)
            if match:
                timestep_numbers.append(int(match.group(1)))

        if timestep_numbers:
            return max(timestep_numbers)  # 가장 큰 timestep 값 반환
        else:
            return -1  # 저장된 데이터가 없으면 -1 반환 (0부터 시작)

    def _on_step(self) -> bool:
        """🔥 PPO 학습 스텝마다 LiDAR 데이터 저장 + Unreal에서 시각화"""
        start_time = time.time()  # 스텝 시작 시간

        # self.locals -> 현재 학습 중인 정보(obs, reward, action 등)
        # stable-baselines3의 BaseCallback 클래스 상속받으면 사용 가능
        reward = self.locals["rewards"][0]  # 현재 스텝의 보상
        obs = self.locals["new_obs"][0]  # 현재 상태 (드론 위치 + LiDAR 데이터)
        action = self.locals["actions"][0]  # 현재 실행된 action
        loss = self.model.logger.name_to_value.get("loss", "N/A")  # PPO 모델의 손실값
        step_in_episode = self.training_env.envs[0].step_in_episode  # 현재 에피소드에서 몇 번째 스텝인지
        current_timesteps = self.num_timesteps  # 현재 스텝 
        lidar_step = self.start_timestep + self.num_timesteps  # ✅ 기존 데이터 이후부터 저장

        # 🔥 LiDAR 데이터 저장 (드론 위치 제외하고 LiDAR 데이터만 저장)
        lidar_points = obs[1:, :]  # (80, 3) shape (첫 번째 행(드론 위치) 제외)
        filename = os.path.join(self.save_path, f"timestep_{lidar_step:06d}.npy")
        np.save(filename, lidar_points)
        print(f"✅ LiDAR 데이터 저장 완료: {filename} (current Timesteps: {lidar_step})")

        # 🔥 Unreal Engine에서 LiDAR 데이터를 실시간 시각화
        # simPlotPoints()는 Unreal Engine의 DrawDebugPoint() 함수를 내부적으로 호출
        self.client.simPlotPoints(
            points=[airsim.Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in lidar_points],  # 변환된 좌표 사용
            color_rgba=[0, 255, 0, 255],  # ✅ 초록색 디버그 포인트
            size=5.0,  # 포인트 크기
            duration=0.2  # 0.2초 후 사라짐
        )
        print(f"📌 {len(lidar_points)}개의 LiDAR 포인트가 Unreal에 표시됨!")

        # ✅ 학습 과정 로그 출력 (스텝당 소요 시간 포함)
        step_duration = time.time() - start_time  # ⏱️ 이번 스텝 소요 시간
        total_duration = time.time() - self.last_time  # ⏱️ 전체 학습 시간

        drone_position = obs[0, :]  # ✅ 드론 위치 (x, y, z)만 가져오기
        print(f"Step {current_timesteps} | Reward: {reward:.2f} | Position: {np.round(drone_position, 2)} | Action: {np.round(action, 2)} | Loss: {loss} || Step in Episode: {step_in_episode}")
        print(f"⏱️ 이번 스텝 소요 시간: {step_duration:.4f}초 | 총 학습 시간: {total_duration:.2f}초\n")

        return True  # ✅ 학습 계속 진행

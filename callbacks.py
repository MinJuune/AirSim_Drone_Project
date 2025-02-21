import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
import time
import open3d as o3d
import airsim

class CustomCallback(BaseCallback):
    def __init__(self, save_path="lidar_data", verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)  # 폴더 생성
        self.last_time = time.time()  # ⏱️ 처음 시간 기록

        # Open3D 시각화 설정
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

    def _on_step(self) -> bool:
        """🔥 PPO 학습 스텝마다 LiDAR 데이터 저장 + 로그 출력"""
        start_time = time.time()  # 스텝 시작 시간

        time.sleep(1)  # 시각화 속도 조절 (0.5초마다 업데이트)

        reward = self.locals["rewards"][0]
        obs = self.locals["new_obs"][0]  # 현재 state (드론 위치 + LiDAR 데이터)
        action = self.locals["actions"][0]
        loss = self.model.logger.name_to_value.get("loss", "N/A")
        step_in_episode = self.training_env.envs[0].step_in_episode
        total_timesteps = self.num_timesteps  # PPO 전체 학습 스텝 수

        # 🔥 LiDAR 데이터 저장 (드론 위치 제외하고 LiDAR 데이터만 저장)
        lidar_points = obs[1:, :]  # (80, 3) shape
        print(f"****{lidar_points}****")
        # filename = os.path.join(self.save_path, f"timestep_{total_timesteps:06d}.npy")
        filename = os.path.join(self.save_path, f"timestep.npy")
        np.save(filename, lidar_points)
        print(f"✅ LiDAR 데이터 저장 완료: {filename} (Total Timesteps: {total_timesteps})")


        # Open3D를 활용한 LiDAR 데이터 실시간 시각화
        # self.pcd.points = o3d.utility.Vector3dVector(lidar_points_from_file)
        # self.pcd.paint_uniform_color([0, 1, 0])  # 초록색
        # self.vis.update_geometry(self.pcd)
        # self.vis.poll_events()
        # self.vis.update_renderer()

        # ✅ 학습 과정 로그 출력 (스텝당 소요 시간 포함)
        step_duration = time.time() - start_time  # ⏱️ 이번 스텝 소요 시간
        total_duration = time.time() - self.last_time  # ⏱️ 전체 학습 시간

        # ✅ 학습 과정 로그 출력
        print(f"Step {total_timesteps} | Reward: {reward:.2f} | Position: {np.round(obs, 2)} | Action: {np.round(action, 2)} | Loss: {loss} || Step in Episode: {step_in_episode}")
        # 좀있다 지워
        print(f"⏱️ 이번 스텝 소요 시간: {step_duration:.4f}초 | 총 학습 시간: {total_duration:.2f}초\n")
        return True



'''

# 📌 시각화할 파일 선택
npy_file = "lidar_data/timestep_000002.npy"  # 원하는 timestep 파일 경로

# 🔥 LiDAR 데이터 로드
lidar_points = np.load(npy_file)  # (100, 3) 형태의 numpy 배열

# ✅ Open3D 포인트 클라우드 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_points)  
pcd.paint_uniform_color([0, 1, 0])  # 초록색 포인트

# 🎨 시각화 실행
o3d.visualization.draw_geometries([pcd])
'''
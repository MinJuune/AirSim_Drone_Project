import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 그래프
from stable_baselines3 import PPO
from envs.airsim_drone_env_0 import AirSimDroneEnv
import keyboard

# MODEL_PATH = "ppo_airsim_drone_policy3.zip"
MODEL_PATH = "weights/ppo_airsim_lidar_sensor.zip"
MAX_EPISODES = 5  # 여러 번 테스트해서 평균 성능 평가
MAX_STEPS = 200  # 한 에피소드에서 최대 스텝 수 (무한 루프 방지)

# False: PPO 자동 조종 / True: 키보드 수동 조종
manual_mode = False 

# 드론 키보드 수동 조종 함수
def manual_control(env):
    client = env.client  # AirSim 환경의 client 가져오기
    print("[INFO] 수동 조종 모드! 키보드로 조작하세요. (X를 누르면 종료)")

    while True:  # 무한 루프 실행 (키 입력 계속 감지)
        if keyboard.is_pressed('w'):
            client.moveByVelocityAsync(1, 0, 0, 1).join()  # 전진
        elif keyboard.is_pressed('s'):
            client.moveByVelocityAsync(-1, 0, 0, 1).join()  # 후진
        elif keyboard.is_pressed('a'):
            client.moveByVelocityAsync(0, -1, 0, 1).join()  # 왼쪽 이동
        elif keyboard.is_pressed('d'):
            client.moveByVelocityAsync(0, 1, 0, 1).join()  # 오른쪽 이동
        elif keyboard.is_pressed('q'):
            client.moveByVelocityAsync(0, 0, -1, 1).join()  # 상승
        elif keyboard.is_pressed('e'):
            client.moveByVelocityAsync(0, 0, 1, 1).join()  # 하강
        elif keyboard.is_pressed('x'):  # X를 누르면 즉시 종료
            print("[INFO] 테스트 종료.")
            env.close()
            exit()  # 모든 실행 종료
        

def test():
    global manual_mode
    env = AirSimDroneEnv()

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] 저장된 모델을 찾을 수 없습니다. 먼저 학습을 진행하세요.")
        return

    print("[INFO] 저장된 모델 로드 중...")
    model = PPO.load(MODEL_PATH, env=env)

    print("[INFO] 테스트 실행 시작...")
    print("[INFO] 'SPACE'를 누르면 수동 조종 모드로 전환됩니다.")
    print("[INFO] 'X'를 누르면 테스트가 종료됩니다.")

    all_positions = []  # 모든 에피소드의 경로 저장
    episode_rewards = []  # 에피소드별 총 보상 저장
    episode_steps = []  # 목표 도달까지 걸린 스텝 수 저장

    for episode in range(MAX_EPISODES):
        obs = env.reset()
        positions_x, positions_y, positions_z = [], [], []
        total_reward = 0

        for step in range(MAX_STEPS):
            if keyboard.is_pressed('space'):
                manual_mode = True  
                print("[WARNING] 수동 조종 모드 활성화! 키보드로 조종하세요!!")
                manual_control(env)  # 수동 조종 시작 (이후 자동 조종 없음)
                return  # 현재 에피소드 종료 (키보드 조작만 허용)

            # PPO 자동 조종
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)

            # 현재 위치 저장
            drone_position = obs[0]  # 상태 배열의 첫 번째 행이 드론의 위치
            x, y, z = drone_position
            positions_x.append(x)
            positions_y.append(y)
            positions_z.append(z)

            total_reward += reward
            print(f"Episode {episode+1} | Step {step} | Reward: {reward:.2f} | Position: (X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f})")

            if done:
                print(f"[INFO] 에피소드 종료")
                break  # 목표 도달하면 바로 종료

        all_positions.append((positions_x, positions_y, positions_z))
        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)

        # 만약 수동모드로 변경된 경우, 에피소드 더이상 진행 안함
        if manual_mode:
            print("수동 모드라 테스트 종료")
            break

    env.close()

    # 테스트 결과 요약
    print("\n[INFO] 테스트 결과 요약")
    print(f"▶ 평균 보상: {np.mean(episode_rewards):.2f}")
    print(f"▶ 평균 목표 도달 스텝 수: {np.mean(episode_steps):.2f}")

    # 3D 그래프 그리기
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, (x_vals, y_vals, z_vals) in enumerate(all_positions):
        ax.plot(x_vals, y_vals, z_vals, marker='o', linestyle='-', label=f'Episode {i+1}')

    # 목표 지점 표시
    target_x, target_y, target_z = env.target_pos
    ax.scatter(target_x, target_y, target_z, color='r', marker='X', s=100, label='Target')

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("3D Flight Path of Drone")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    test()

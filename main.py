import setup_path
import airsim
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from envs.airsim_drone_env_0 import AirSimDroneEnv
from callbacks import CustomCallback  # 콜백 파일 import
from model.PPO_train import train_ppo

def main():
    """메인 실행 함수"""
    env = AirSimDroneEnv()  # 환경 생성

    try:
        model = train_ppo(env)

    except Exception as e:
        print(f"[ERROR] 실행 중 오류 발생: {e}")

    finally:
        env.close()  # 환경 종료
        print("[INFO] 실행 종료 및 환경 닫기 완료.")


if __name__ == "__main__":
    main()
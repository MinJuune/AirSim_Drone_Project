from stable_baselines3 import PPO
from callbacks import CustomCallback
import os
import config
import traceback

def train_ppo(env):
    """PPO 모델 학습 함수"""
    try:
        
        if os.path.exists(config.MODEL_PATH):
            print("[INFO] 기존 모델 로드 중...")
            model = PPO.load(config.MODEL_PATH, env=env)
        else:
            print("[INFO] 새로운 모델 생성...")
            # 저장할 폴더가 없으면 생성
            os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
            
            model = PPO(config.PPO_POLICY, env, verbose=1, 
                        n_steps=config.PPO_N_STEPS, 
                        batch_size=config.PPO_BATCH_SIZE, 
                        n_epochs=config.PPO_N_EPOCHS)

        print("[INFO] 모델 학습 시작...")
        model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=CustomCallback())  # ✅ 콜백 사용
        model.save(config.MODEL_PATH)
        print(f"[INFO] 모델 저장 완료: {config.MODEL_PATH}")
       
    

    except Exception as e:
        print(f"[ERROR] PPO 학습 중 오류 발생: {e}")
        traceback.print_exc()

    return model  # ✅ 학습된 모델 반환
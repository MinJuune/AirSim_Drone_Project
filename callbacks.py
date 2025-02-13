import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CustomCallback(BaseCallback):
    def _on_step(self) -> bool:
        """학습 과정 중 로그 출력"""
        reward = self.locals["rewards"][0]
        obs = self.locals["new_obs"][0]
        action = self.locals["actions"][0]
        loss = self.model.logger.name_to_value.get("loss", "N/A")

        step_in_episode=self.training_env.envs[0].step_in_episode

        print(f"Step {self.num_timesteps} | Reward: {reward:.2f} | Position: {np.round(obs, 2)} | Action: {np.round(action, 2)} | Loss: {loss} || Step in Episode: {step_in_episode}")
        return True


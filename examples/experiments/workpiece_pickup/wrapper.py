import numpy as np
import requests
import gymnasium as gym
import time
from franka_env.envs.franka_env import FrankaEnv


class WorkpiecePickupEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        self._send_gripper_command(1.0)

        # Lift vertically before the lateral reset move to reduce collision risk
        # when the previous episode ended near clutter.
        safe_pose = self.currpos.copy()
        safe_pose[2] = max(self.currpos[2], self.resetpos[2]) + 0.03
        self.interpolate_move(safe_pose, timeout=0.6)

        obs, info = super().reset(**kwargs)
        self._send_gripper_command(1.0)
        time.sleep(max(0.5, self.gripper_sleep))
        self._update_currpos()
        obs = self._get_obs()
        return obs, info


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    @staticmethod
    def _extract_gripper_pos(observation, info=None):
        if info and "original_state_obs" in info:
            return float(np.asarray(info["original_state_obs"]["gripper_pose"]).reshape(-1)[0])
        return float(np.asarray(observation["state"]).reshape(observation["state"].shape[0], -1)[0, -1])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self._extract_gripper_pos(obs, info)
        return obs, info

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]

        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.9
        ):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = self._extract_gripper_pos(observation, info)
        return observation, reward, terminated, truncated, info

import os
import jax
import numpy as np
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import (
    binary_classifier_probability,
    load_classifier_func,
)

from experiments.config import DefaultTrainingConfig
from experiments.workpiece_pickup.wrapper import (
    GripperPenaltyWrapper,
    WorkpiecePickupEnv,
)


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://localhost:5000/"
    # Measured workspace corners from bringup/pose.log:
    # x in [0.40768, 0.65181], y in [-0.43204, 0.09863].
    # We keep a small inset from the measured extremes to leave margin for
    # calibration error and avoid scraping the bin edge during exploration.
    WORKSPACE_XY_LOW = np.array([0.41568, -0.42404])
    WORKSPACE_XY_HIGH = np.array([0.64381, 0.09063])
    WORKSPACE_CENTER_XY = (WORKSPACE_XY_LOW + WORKSPACE_XY_HIGH) / 2

    REALSENSE_CAMERAS = {
        "wrist": {
            "serial_number": "335122272207",
            "dim": (1280, 720),
            "exposure": 20000,
        },
        "side_policy": {
            "serial_number": "233522079237",
            "dim": (1280, 720),
            "exposure": 15000,
        },
        "side_classifier": {
            "serial_number": "233522079237",
            "dim": (1280, 720),
            "exposure": 15000,
        },
    }
    IMAGE_CROP = {
        "wrist": lambda img: img[:, :],
        "side_policy": lambda img: img[360:560, 450:900],
        "side_classifier": lambda img: img[360:560, 450:900],
    }

    RESET_POSE = np.array(
        [
            0.47962504594521777,
            -0.1606138124342781,
            0.15,
            -3.0802704821179843,
            -0.04013589859939137,
            -0.06952010716295809,
        ]
    )

    # This task uses a learned reward classifier instead of a pose reward, so
    # TARGET_POSE acts as a nominal hover pose rather than a success pose.
    TARGET_POSE = RESET_POSE - np.array([0, 0, 0.1, 0, 0, 0])

    ACTION_SCALE = np.array([0.015, 0.1, 1])
    RANDOM_RESET = True
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.035
    RANDOM_RZ_RANGE = 0.08
    ABS_POSE_LIMIT_LOW = np.array(
        [
            WORKSPACE_XY_LOW[0],
            WORKSPACE_XY_LOW[1],
            0.010,
            np.pi - 0.15,
            -0.12,
            -np.pi / 2,
        ]
    )
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            WORKSPACE_XY_HIGH[0],
            WORKSPACE_XY_HIGH[1],
            0.255,
            np.pi + 0.10,
            0.12,
            np.pi / 2,
        ]
    )
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.006,
        "translational_clip_y": 0.0059,
        "translational_clip_z": 0.0035,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.0035,
        "rotational_clip_x": 0.02,
        "rotational_clip_y": 0.02,
        "rotational_clip_z": 0.015,
        "rotational_clip_neg_x": 0.02,
        "rotational_clip_neg_y": 0.02,
        "rotational_clip_neg_z": 0.015,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }
    MAX_EPISODE_LENGTH = 200


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["side_policy", "wrist"]
    classifier_keys = ["side_classifier"]
    classifier_reward_threshold = 0.8
    classifier_min_episode_steps = 30
    classifier_success_streak = 3
    # This task uses two camera streams plus a learned gripper head; the
    # default batch size of 256 exhausts an 8GB GPU during learner updates.
    batch_size = 128
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = WorkpiecePickupEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                return int(
                    binary_classifier_probability(classifier, obs)
                    > self.classifier_reward_threshold
                )

            env = MultiCameraBinaryRewardClassifierWrapper(
                env,
                reward_func,
                min_episode_steps=self.classifier_min_episode_steps,
                success_streak_needed=self.classifier_success_streak,
            )
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env

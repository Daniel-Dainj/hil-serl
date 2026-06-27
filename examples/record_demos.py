import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer(
    "min_episode_steps",
    100,
    "Minimum steps before classifier success is allowed to finish a demo.",
)
flags.DEFINE_integer(
    "success_streak_needed",
    10,
    "Number of consecutive successful classifier steps required before reset.",
)


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=True)

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    episode_steps = 0
    success_streak = 0

    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape)
        next_obs, rew, done, truncated, info = env.step(actions)
        episode_steps += 1
        raw_success = bool(info.get("succeed", False))
        if raw_success:
            success_streak += 1
        else:
            success_streak = 0

        accept_success = (
            raw_success and episode_steps >= FLAGS.min_episode_steps and success_streak >= FLAGS.success_streak_needed
        )
        if raw_success and not accept_success and not info.get("env_done", False):
            info = dict(info)
            info["classifier_succeed"] = True
            info["succeed"] = False
            rew = info.get("env_reward", 0)
            done = False

        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)

        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            trajectory = []
            returns = 0
            episode_steps = 0
            success_streak = 0
            obs, info = env.reset()

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")


if __name__ == "__main__":
    app.run(main)

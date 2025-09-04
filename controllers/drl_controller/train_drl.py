import os, glob
from datetime import datetime
import pandas as pd, matplotlib.pyplot as plt
import numpy as np
from webots_gym_env import WebotsGymnasiumEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure

MAX_SPEED = 6.4
GOAL_REWARD = 50
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = "./checkpoints/"
MONITOR_LOG_DIR = "./monitor_logs/"
SUMMARY_CSV = "training_summary_log.csv"
TIMESTEPS_TO_TRAIN = 4000000

def find_latest_checkpoint():
    ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "ppo_webots_*_steps.zip"))
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split("_")[-2]))
        return ckpts[-1]
    return None

def get_next_training_round(csv_path=SUMMARY_CSV):
    if not os.path.exists(csv_path): return 1
    df = pd.read_csv(csv_path, comment="#")
    return int(df["training_round"].max()) + 1 if "training_round" in df.columns else 1

def get_next_training_index(csv_path=SUMMARY_CSV):
    if not os.path.exists(csv_path): return 1
    with open(csv_path, "r") as f:
        lines = f.readlines()
    return sum(1 for line in lines if line.startswith("# =====")) + 1

def append_training_summary_log(new_rows, csv_path=SUMMARY_CSV):
    dir_name = os.path.dirname(csv_path)
    if dir_name: os.makedirs(dir_name, exist_ok=True)
    is_new_file = not os.path.exists(csv_path)
    if not is_new_file:
        with open(csv_path, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n# ===== New Training Session @ {timestamp} =====\n")
    pd.DataFrame(new_rows).to_csv(csv_path, mode="a", index=False, header=is_new_file)

class OutcomeLoggingCallback(BaseCallback):
    def __init__(self): super().__init__(); self.episode_info = []
    def _on_step(self):
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][i]
                    self.episode_info.append(info.copy())
        return True

def plot_episode_reward_curve(summary_csv=SUMMARY_CSV, log_dir=MONITOR_LOG_DIR,
                              training_index=None, rolling=25):
    if not os.path.exists(summary_csv): return
    df = pd.read_csv(summary_csv, comment="#")
    df = df[df["training_round"] == TRAINING_ROUND]
    if df.empty: return

    plt.figure(figsize=(12, 5))
    plt.plot(df["episode"], df["reward"], label="Episode Reward", linewidth=1)
    if rolling and len(df) >= 2:
        plt.plot(df["episode"], df["reward"].rolling(rolling, min_periods=1).mean(),
                 label=f"Rolling Mean ({rolling})", linewidth=2)
    plt.title(f"Episode Reward Curve (Training Round {TRAINING_ROUND})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    os.makedirs(log_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"Training_{training_index:02d}_episode_reward.png"))
    plt.close()

def plot_success_rate_per_evaluation(summary_csv=SUMMARY_CSV, log_dir=MONITOR_LOG_DIR,
                                     training_index=None, episodes_per_eval=5):
    if not os.path.exists(summary_csv): return
    df = pd.read_csv(summary_csv, comment="#")
    df = df[df["training_round"] == TRAINING_ROUND]
    if df.empty: return

    df["eval_bin"] = (df["episode"] - 1) // episodes_per_eval
    agg = (df.groupby("eval_bin")["outcome"]
             .apply(lambda s: (s == "goal").mean() * 100)
             .reset_index(name="success_rate"))

    plt.figure(figsize=(12, 5))
    plt.plot(agg["eval_bin"], agg["success_rate"], marker="o", linewidth=2, label="Success Rate (%)")
    plt.title(f"Success Rate per Evaluation (every {episodes_per_eval} episodes) — Round {TRAINING_ROUND}")
    plt.xlabel("Evaluation Index")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)
    plt.ylim(0, 100)
    plt.legend()
    os.makedirs(log_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"Training_{training_index:02d}_success_rate.png"))
    plt.close()

TRAINING_ROUND = get_next_training_round()

if __name__ == "__main__":
    os.makedirs(MONITOR_LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    raw_env = WebotsGymnasiumEnv()
    monitored_env = Monitor(raw_env, filename="./monitor_logs/monitor.csv")
    vec_env = DummyVecEnv([lambda: monitored_env])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)


    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=CHECKPOINT_DIR, name_prefix="ppo_webots")
    outcome_logger = OutcomeLoggingCallback()


    logger = configure("./logs/", ["stdout", "csv", "tensorboard"])


    latest = find_latest_checkpoint()
    if latest:
        print("Resuming from:", latest)
        model = PPO.load(latest, env=env)
        model.set_logger(logger)
        model.learn(total_timesteps=TIMESTEPS_TO_TRAIN,
                    callback=[checkpoint_callback, outcome_logger],
                    reset_num_timesteps=False)
    else:
        print("Starting training from scratch")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=LEARNING_RATE,
            tensorboard_log="./logs/",
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        model.set_logger(logger)
        model.learn(total_timesteps=TIMESTEPS_TO_TRAIN,
                    callback=[checkpoint_callback, outcome_logger])

    env.save("vecnormalize.pkl")
    env.close()

    monitor_csv = os.path.join(MONITOR_LOG_DIR, "monitor.csv")
    if os.path.exists(monitor_csv):

        df = pd.read_csv(monitor_csv, comment="#", header=None, names=["r", "l", "t"])

        rewards = pd.to_numeric(df["r"], errors="coerce")
        steps = pd.to_numeric(df["l"], errors="coerce")

        summary = pd.DataFrame({
            "steps": steps.fillna(0).astype(int),
            "reward": rewards
        }).reset_index(drop=True)

        summary["episode"] = np.arange(1, len(summary) + 1)
        summary["training_round"] = TRAINING_ROUND

        def classify(row, info_list):
            idx = int(row["episode"]) - 1
            if 0 <= idx < len(info_list) and isinstance(info_list[idx], dict) and "done_reason" in info_list[idx]:
                return info_list[idx]["done_reason"]
            reward = row["reward"]
            if pd.isna(reward):
                return "timeout"
            if reward >= GOAL_REWARD:
                return "goal"
            elif reward < 0:
                return "crash"
            return "timeout"

        summary["outcome"] = summary.apply(lambda row: classify(row, outcome_logger.episode_info), axis=1)

        summary = summary[["episode", "training_round", "outcome", "steps", "reward"]]
        append_training_summary_log(summary)

    training_index = get_next_training_index()
    plot_episode_reward_curve(training_index=training_index, rolling=25)
    plot_success_rate_per_evaluation(training_index=training_index, episodes_per_eval=5)
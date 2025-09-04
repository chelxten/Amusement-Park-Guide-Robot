import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- USER PATHS ---
MONITOR_CSV = "monitor.csv"                       # path to your monitor file
SUMMARY_CSV = "training_summary_log.csv"        # path to your per-episode summary (with 'outcome')
EVAL_CSV    = None                                # set to e.g. "runs/run_014/eval.csv" if you have it

# --- SETTINGS ---
ROLL_WINDOW_REWARD = 25        # smoothing for reward curve
EPISODES_PER_EVAL  = 10        # used only if EVAL_CSV is None (proxy eval window)

FIGDIR = "figures_from_logs"
os.makedirs(FIGDIR, exist_ok=True)

# ---------- 1) EPISODE REWARD CURVE (from monitor.csv) ----------
def load_monitor_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"monitor.csv not found at: {path}")
    # SB3 monitor files may have a JSON header commented with '#'
    df = pd.read_csv(path, comment="#")
    # Normalize column names and fallback if needed
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "r" not in df.columns or "l" not in df.columns:
        # Tolerate odd headers (first two cols are reward & length)
        rename = {df.columns[0]: "r", df.columns[1]: "l"}
        if len(df.columns) >= 3:
            rename[df.columns[2]] = "t"
        df = df.rename(columns=rename)
    # Coerce numeric
    for c in ("r","l","t"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop any rows without reward/length
    df = df.dropna(subset=["r","l"]).reset_index(drop=True)
    return df

mon = load_monitor_csv(MONITOR_CSV)
mon["episode"] = np.arange(1, len(mon)+1)
roll = mon["r"].rolling(window=ROLL_WINDOW_REWARD, min_periods=1).mean()

plt.figure(figsize=(12,5))
plt.plot(mon["episode"], mon["r"], alpha=0.25, label="Episode Reward")
plt.plot(mon["episode"], roll, linewidth=2.0, label=f"Rolling Mean ({ROLL_WINDOW_REWARD})")
plt.title("Episode Reward Curve")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "episode_reward_curve.png"), dpi=300)
plt.close()

# ---------- 2) SUCCESS RATE PER EVALUATION ----------
def plot_success_from_eval_csv(path: str):
    df = pd.read_csv(path)
    if not {"eval_index","success_rate"}.issubset(df.columns):
        raise ValueError(f"'eval.csv' missing required columns in {path}")
    df = df.sort_values("eval_index")
    plt.figure(figsize=(12,5))
    plt.plot(df["eval_index"], df["success_rate"]*100.0, marker="o", linewidth=1.5, label="Success Rate (%)")
    plt.ylim(0, 100)
    plt.xlabel("Evaluation Index"); plt.ylabel("Success Rate (%)")
    plt.title("Success Rate per Evaluation")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "success_rate_per_evaluation.png"), dpi=300)
    plt.close()

def plot_success_from_summary_csv(path: str, episodes_per_eval: int):
    # Accept commas or whitespace as separators
    df = pd.read_csv(path, sep=r"[,\s]+", engine="python", header=None)
    # Try to detect columns; expect: episode, training_round, outcome, steps, reward
    # If there are 5 columns, map directly; if 4 (missing reward/steps), be lenient.
    if df.shape[1] >= 5:
        df = df.iloc[:, :5]
        df.columns = ["episode", "training_round", "outcome", "steps", "reward"]
    elif df.shape[1] == 4:
        df.columns = ["episode", "training_round", "outcome", "steps"]
        df["reward"] = np.nan
    else:
        # fallback: try to read with header row present
        df = pd.read_csv(path, sep=r"[,\s]+", engine="python")
        # ensure required columns exist
        for name in ["episode","training_round","outcome"]:
            if name not in df.columns:
                raise ValueError(f"Could not parse {path}; expected 'episode, training_round, outcome, ...'")

    # Clean types
    for c in ["episode","training_round","steps","reward"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "outcome" in df.columns:
        df["outcome"] = df["outcome"].astype(str).str.strip().str.lower()
    else:
        raise ValueError("No 'outcome' column found; cannot estimate success rate.")

    # Proxy evaluation windows across training episodes
    df = df.sort_values("episode").reset_index(drop=True)
    df["eval_index"] = ((df["episode"] - 1) // episodes_per_eval) + 1
    grp = df.groupby("eval_index")
    succ_rate = (grp.apply(lambda g: (g["outcome"] == "goal").mean() * 100.0)
                 .reset_index(name="success_rate_pct"))

    plt.figure(figsize=(12,5))
    plt.plot(succ_rate["eval_index"], succ_rate["success_rate_pct"], marker="o", linewidth=1.5, label="Success Rate (%)")
    plt.ylim(0, 100)
    plt.xlabel("Evaluation Index (proxy windows of episodes)")
    plt.ylabel("Success Rate (%)")
    plt.title(f"Success Rate per Evaluation (proxy: every {episodes_per_eval} episodes)")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "success_rate_per_evaluation.png"), dpi=300)
    plt.close()

if EVAL_CSV and os.path.exists(EVAL_CSV):
    plot_success_from_eval_csv(EVAL_CSV)
else:
    # Use your per-episode summary (with 'outcome') as a proxy
    plot_success_from_summary_csv(SUMMARY_CSV, EPISODES_PER_EVAL)

print(f"✅ Saved plots in: {os.path.abspath(FIGDIR)}")
print(" - episode_reward_curve.png")
print(" - success_rate_per_evaluation.png")
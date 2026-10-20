"""
Two-panel IEEE figure:
  Panel A: PPO_5 training reward (smoothed) vs timesteps
  Panel B: tiltfix2 checkpoint evaluation vs checkpoint step

Run from ~/projects/AGERE/
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ------------------------------------------------------------------
TB_RUN = "PPO_5"          # Panel A source
FAMILY = "tiltfix2"       # Panel B family: tiltfix | tiltfix2 | tiltfix3
SMOOTH_WINDOW = 50        # rolling-mean window for Panel A
# ------------------------------------------------------------------

# --- Panel A data ---
tb = pd.read_csv("tb_all_scalars.csv")
rew = (tb[(tb["run"] == TB_RUN) & (tb["tag"] == "rollout/ep_rew_mean")]
       .sort_values("step"))
rew_smooth = rew["value"].rolling(SMOOTH_WINDOW, min_periods=1).mean()

# --- Panel B data ---
sweep = pd.read_csv("results/checkpoints_ranked/sweep_summary.csv")
sweep["family"] = sweep["path"].str.extract(r"disturbance_3x5_(tiltfix\d*)_")
sweep["step"]   = sweep["step_label"].str.extract(r"(\d+)_steps").astype(int)
fam = sweep[sweep["family"] == FAMILY].sort_values("step").reset_index(drop=True)
assert not fam.empty, f"No rows for family={FAMILY} in sweep_summary.csv"

# --- Figure ---
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
})
fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(3.5, 4.2), dpi=300,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.45},
)

# ---- Panel A ----
axA.plot(rew["step"], rew["value"], lw=0.6, alpha=0.25, color="tab:blue")
axA.plot(rew["step"], rew_smooth, lw=1.4, color="tab:blue")
axA.set_xlabel("Training timesteps")
axA.set_ylabel("Episode reward")
axA.set_xlim(0, 520_000)
axA.grid(alpha=0.3, lw=0.4)
axA.set_title(f"(a) PPO training reward — {TB_RUN}", loc="left")

# ---- Panel B ----
axB.plot(fam["step"], fam["mean_pos_error"], "o-", color="tab:red",
         ms=3, lw=1.2, label="Mean pos. error")
axB.set_xlabel("Checkpoint step")
axB.set_ylabel("Mean position error (m)", color="tab:red")
axB.tick_params(axis="y", labelcolor="tab:red")
axB.set_xlim(0, 520_000)
axB.grid(alpha=0.3, lw=0.4)
axB.set_title(f"(b) {FAMILY} checkpoint evaluation", loc="left")

axB2 = axB.twinx()
axB2.plot(fam["step"], fam["crash_rate"],          "s--", color="black",
          ms=3, lw=1.0, label="Overall crash")
axB2.plot(fam["step"], fam["kick_crash_rate"],     "^--", color="tab:orange",
          ms=3, lw=1.0, label="Kick crash")
axB2.plot(fam["step"], fam["torque_crash_rate"],   "v--", color="tab:purple",
          ms=3, lw=1.0, label="Torque crash")
axB2.plot(fam["step"], fam["wind_crash_rate"],     "o--", color="tab:green",
          ms=3, lw=1.0, label="Wind crash")
axB2.yaxis.set_major_formatter(PercentFormatter(1.0))
axB2.set_ylabel("Crash rate")
axB2.set_ylim(0, 0.6)

# Combined legend for Panel B
lines  = axB.get_lines() + axB2.get_lines()
labels = [l.get_label() for l in lines]
axB.legend(lines, labels, loc="upper center", ncol=2,
           frameon=False, fontsize=6)

plt.tight_layout()
plt.savefig("fig_training.pdf", bbox_inches="tight")
plt.savefig("fig_training.png", dpi=300, bbox_inches="tight")
print("wrote fig_training.pdf and fig_training.png")
print("\nPanel B table (for your sanity check):")
print(fam[["step", "mean_pos_error", "crash_rate",
           "kick_crash_rate", "torque_crash_rate", "wind_crash_rate"]]
      .to_string(index=False))

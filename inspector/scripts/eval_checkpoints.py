"""
Step-two checkpoint evaluation (inspector module).

Evaluates two groups of checkpoints under ONE frozen protocol:

    chain       the cumulative warm-started line: disturbance_3x5, ..._tiltfix, ..._tiltfix2, ...
                (intermediate checkpoints + the final save of every line). One selection unit.
    seedstudy   the 10 from-scratch training seeds (intermediate checkpoints + final save).
                One selection unit PER training seed.

Pipeline:
    init    freeze protocol + inventory (hash, timesteps, hyperparameters)
    screen  every intermediate checkpoint, cheap balanced grid      (stage "screen")
    select  top-K of each unit, thorough balanced grid               (stage "select")
    choose  champion per unit from "select" data only -> frozen      (summaries/champions.csv)
    report  held-out balanced grid on a fresh seed block             (stage "report")
    combine tidy per-stage tables + summaries (Wilson 95% CIs)

Rules:
  * Balanced grid: episodes_per_cell for each of the 15 (type, level) cells, fixed seed per cell.
    Every checkpoint -- in both groups -- sees the identical start/disturbance draws (all randomness is
    drawn in HoverGymEnv.reset(), independent of the policy), so comparisons are paired.
  * screen / select / report use disjoint base seeds. `report` refuses to run before `choose`.
  * The protocol is written once by `init`; every later command reads it.
  * One shard CSV per (stage, checkpoint); every command is resumable (existing shards are skipped).
  * Final saves are baseline arms, never selection candidates.

Run from the AGERE repo root:
    python inspector/scripts/eval_checkpoints.py init    --out results/evaluation/step2
    python inspector/scripts/eval_checkpoints.py screen  --out results/evaluation/step2 --workers 12
    python inspector/scripts/eval_checkpoints.py select  --out results/evaluation/step2 --workers 12
    python inspector/scripts/eval_checkpoints.py choose  --out results/evaluation/step2
    python inspector/scripts/eval_checkpoints.py report  --out results/evaluation/step2 --workers 12
    python inspector/scripts/eval_checkpoints.py combine --out results/evaluation/step2

Check what `init` found (inventory.csv, printed counts) before running screen.
Needs pandas for `combine` only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

# Make `src` importable when run as a plain script from the repo root (also in spawned workers).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from src.config import DISTURBANCE_LEVELS, DISTURBANCE_TYPES  # noqa: E402
from src.paths import MODEL_WEIGHTS_DIR  # noqa: E402

CELLS = [(t, lvl) for t in DISTURBANCE_TYPES for lvl in range(1, DISTURBANCE_LEVELS + 1)]

EPISODE_FIELDS = [
    "ckpt_id", "group", "unit", "line", "training_seed", "role", "ckpt_step", "stage", "eval_seed",
    "cell_type", "cell_level", "episode_idx", "disturbance_fired", "magnitude",
    "crash", "truncation_reason", "n_steps", "recovered", "recovery_time_steps",
    "final_pos_error", "mean_pos_error_episode", "max_pos_error_episode",
    "peak_tilt_rad", "total_reward", "wind_ss_error",
]

INVENTORY_FIELDS = [
    "ckpt_id", "group", "unit", "line", "training_seed", "role", "ckpt_step", "file_step", "path",
    "sha256", "gamma", "ent_coef", "learning_rate", "n_steps_rollout",
]

README = """# Step-two checkpoint evaluation

Layout
- protocol.json          frozen settings (seeds, episode counts, selection rule, git commit, task config, versions)
- inventory.csv          one row per checkpoint: id, group, unit, line, training seed, role, steps, path, sha256, hyperparameters
- episodes/{stage}/      one shard CSV per checkpoint (resumable work units)
- episodes/{stage}.csv   tidy long-format table, one row per episode (made by `combine`)
- summaries/             screen_ranking.csv, select_candidates.csv, champions.csv,
                         checkpoint_summary.csv, cell_summary.csv, champion_report.csv

Groups / units
- group chain      : cumulative warm-started line (disturbance_3x5 -> tiltfix -> tiltfix2 -> ...). unit = "chain".
- group seedstudy  : 10 from-scratch seeds. unit = "seedstudy_s<seed>".
- line             : training tag of the run (chain) or the seedstudy tag.
- role             : candidate (intermediate checkpoint) | final (end-of-run save; baseline arm, never a candidate).
- ckpt_step        : model.num_timesteps stored in the zip. file_step: step number in the filename.

Stages: screen -> select -> (choose) -> report. Champions are chosen from `select` data only;
`report` uses a fresh seed block and is never used for selection.

Episode columns: ckpt_id, group, unit, line, training_seed, role, ckpt_step, stage, eval_seed (base seed of the cell),
cell_type, cell_level, episode_idx (within cell), disturbance_fired, magnitude, crash (0/1),
truncation_reason, n_steps, recovered, recovery_time_steps, final_pos_error (m),
mean_pos_error_episode (m, mean over steps), max_pos_error_episode (m), peak_tilt_rad,
total_reward, wind_ss_error (m, wind only).

CIs are Wilson 95% intervals treating episodes as independent. For claims about the training METHOD the
unit of replication is the training seed (seedstudy), not the episode.
"""


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------
def default_protocol(args, seeds: list[int], study_name: str) -> dict:
    return {
        "study_name": study_name,
        "stage_preset": "disturbance_3x5",
        "groups": {
            "chain": {"enabled": not args.no_chain, "training_seed": args.chain_seed,
                      "line_regex": args.chain_line},
            "seedstudy": {"enabled": not args.no_seedstudy, "tag": args.seedstudy_tag,
                          "training_seeds": seeds,
                          "expected_hparams": {"gamma": 0.995, "ent_coef": 0.003},
                          "expected_candidates_per_seed": 10},
        },
        "screen": {"episodes_per_cell": 4, "base_seed": 10_000,
                   "top_k": {"chain": args.top_k_chain, "seedstudy": args.top_k_seedstudy}},
        "select": {"episodes_per_cell": 15, "base_seed": 20_000},
        "report": {"episodes_per_cell": 20, "base_seed": 30_000, "scope": args.report_scope},
        "selection_rule": (
            "Per unit (chain = one unit; seedstudy = one unit per training seed). Stage screen: rank the "
            "intermediate (candidate) checkpoints by overall crash rate on the balanced screen grid "
            "(tie-break: lower mean final position error, then later step); keep top_k. Stage select: "
            "evaluate those on the balanced select grid; champion = lowest crash rate (same tie-breaks). "
            "Final saves are baseline arms, never candidates. The report stage uses a fresh seed block "
            "and is not used for any selection. Every checkpoint is evaluated with the CURRENT task "
            "config (stage preset disturbance_3x5), recorded in this file."
        ),
    }


def load_protocol(out: Path) -> dict:
    p = out / "protocol.json"
    if not p.exists():
        sys.exit(f"No protocol at {p}. Run `init` first.")
    return json.loads(p.read_text())


def parse_seeds(s: str) -> list[int]:
    if "-" in s and "," not in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def _git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    status = run("git", "status", "--porcelain")
    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "torch", "stable_baselines3", "gymnasium", "pybullet"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = None
    return out


# ----------------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inv_row(PPO, p: Path, *, group, unit, line, seed, role, file_step) -> dict:
    model = PPO.load(str(p), device="cpu")
    lr = model.learning_rate
    return {
        "ckpt_id": p.stem.replace("hover_stabilize_ppo_", ""),
        "group": group, "unit": unit, "line": line, "training_seed": seed, "role": role,
        "ckpt_step": int(model.num_timesteps), "file_step": file_step, "path": str(p),
        "sha256": _sha256(p), "gamma": float(model.gamma), "ent_coef": float(model.ent_coef),
        "learning_rate": float(lr) if not callable(lr) else "schedule",
        "n_steps_rollout": int(model.n_steps),
    }


def build_inventory(proto: dict) -> list[dict]:
    from stable_baselines3 import PPO

    ckpt_dir = MODEL_WEIGHTS_DIR / "checkpoints"
    rows: list[dict] = []

    g = proto["groups"]["chain"]
    if g["enabled"]:
        s, line_re = g["training_seed"], g["line_regex"]
        mid = re.compile(rf"^hover_stabilize_ppo_seed{s}_({line_re})_(\d+)_steps\.zip$")
        fin = re.compile(rf"^hover_stabilize_ppo_seed{s}_({line_re})\.zip$")
        n_c = n_f = 0
        for p in sorted(ckpt_dir.glob("*.zip")):
            m = mid.match(p.name)
            if m:
                rows.append(_inv_row(PPO, p, group="chain", unit="chain", line=m.group(1), seed=s,
                                     role="candidate", file_step=int(m.group(2))))
                n_c += 1
        for p in sorted(MODEL_WEIGHTS_DIR.glob("*.zip")):
            m = fin.match(p.name)
            if m:
                rows.append(_inv_row(PPO, p, group="chain", unit="chain", line=m.group(1), seed=s,
                                     role="final", file_step=-1))
                n_f += 1
        lines = sorted({r["line"] for r in rows if r["group"] == "chain"})
        print(f"[chain] {n_c} candidates, {n_f} final saves, lines: {lines}")

    g = proto["groups"]["seedstudy"]
    if g["enabled"]:
        tag = g["tag"]
        for seed in g["training_seeds"]:
            pat = re.compile(rf"^hover_stabilize_ppo_seed{seed}_{re.escape(tag)}_(\d+)_steps\.zip$")
            paths = [(p, int(pat.match(p.name).group(1))) for p in sorted(ckpt_dir.glob("*.zip"))
                     if pat.match(p.name)]
            for p, fs in paths:
                rows.append(_inv_row(PPO, p, group="seedstudy", unit=f"seedstudy_s{seed}", line=tag,
                                     seed=seed, role="candidate", file_step=fs))
            final = MODEL_WEIGHTS_DIR / f"hover_stabilize_ppo_seed{seed}_{tag}.zip"
            if final.exists():
                rows.append(_inv_row(PPO, final, group="seedstudy", unit=f"seedstudy_s{seed}", line=tag,
                                     seed=seed, role="final", file_step=-1))
            else:
                print(f"WARNING seedstudy seed {seed}: final save {final.name} not found")
            if len(paths) != g["expected_candidates_per_seed"]:
                print(f"WARNING seedstudy seed {seed}: found {len(paths)} candidates, expected "
                      f"{g['expected_candidates_per_seed']}")
        print(f"[seedstudy] {len([r for r in rows if r['group'] == 'seedstudy'])} checkpoints "
              f"across {len(g['training_seeds'])} seeds")

    ids = [r["ckpt_id"] for r in rows]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        sys.exit(f"Duplicate checkpoint ids (filenames collide across groups): {sorted(dup)}")
    return rows


def read_inventory(out: Path) -> list[dict]:
    p = out / "inventory.csv"
    if not p.exists():
        sys.exit(f"No inventory at {p}. Run `init` first.")
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["training_seed"] = int(r["training_seed"])
        r["ckpt_step"] = int(r["ckpt_step"])
    return rows


def cmd_init(args) -> None:
    out = Path(args.out)
    if (out / "protocol.json").exists() and not args.force:
        sys.exit(f"{out}/protocol.json exists. The protocol is frozen; use --force only if you are "
                 f"deliberately starting a new study in the same folder.")
    from src.config import HOVER_STAGE_PRESETS, ProjectConfig

    proto = default_protocol(args, parse_seeds(args.seedstudy_seeds), out.name)
    config = ProjectConfig()
    config.task = replace(config.task, **HOVER_STAGE_PRESETS[proto["stage_preset"]])
    proto["task_config"] = asdict(config.task)
    proto["git"] = _git_info()
    proto["versions"] = _versions()
    proto["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

    inv = build_inventory(proto)
    if not inv:
        sys.exit("No checkpoints found. Check MODEL_WEIGHTS_DIR, --chain-line and --seedstudy-tag.")

    exp = proto["groups"]["seedstudy"]["expected_hparams"]
    bad = [r for r in inv if r["group"] == "seedstudy"
           and (abs(r["gamma"] - exp["gamma"]) > 1e-9 or abs(r["ent_coef"] - exp["ent_coef"]) > 1e-9)]
    if bad:
        print("HYPERPARAMETER MISMATCH in seedstudy (expected %s):" % exp)
        for r in bad:
            print(f"  {r['ckpt_id']}: gamma={r['gamma']} ent_coef={r['ent_coef']}")
        sys.exit("Aborting: the seeds were not all trained with the frozen recipe.")

    for sub in ("episodes", "summaries"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(proto, indent=2))
    with open(out / "inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INVENTORY_FIELDS)
        w.writeheader()
        w.writerows(inv)
    (out / "README.md").write_text(README)
    print(f"Wrote protocol.json and inventory.csv ({len(inv)} checkpoints) to {out}")


# ----------------------------------------------------------------------------
# Evaluation worker (spawned process; one PyBullet client per worker)
# ----------------------------------------------------------------------------
def _eval_grid_worker(job: dict) -> str:
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    from stable_baselines3 import PPO
    from src.config import HOVER_STAGE_PRESETS, ProjectConfig
    from src.training.evaluate.hover_evaluate import run_episode
    from src.training.gym_wrapper.hover_gym_wrapper import HoverGymEnv

    config = ProjectConfig()
    config.task = replace(config.task, **HOVER_STAGE_PRESETS[job["stage_preset"]])
    env = HoverGymEnv(config)
    model = PPO.load(job["path"], device="cpu")
    c = job["ckpt"]
    rows = []
    for ci, (t, lvl) in enumerate(CELLS):
        # _forced_disturbance is read fresh in every reset(): it pins type/level while
        # magnitude, direction and onset are still drawn from the seeded stream.
        env._forced_disturbance = {"type": t, "level": lvl}
        cell_seed = job["base_seed"] + ci
        for ep in range(job["episodes_per_cell"]):
            res = run_episode(env, model, seed=cell_seed if ep == 0 else None)
            tr, tl = res["pos_error_trace"], res["tilt_trace"]
            rows.append({
                "ckpt_id": c["ckpt_id"], "group": c["group"], "unit": c["unit"], "line": c["line"],
                "training_seed": c["training_seed"], "role": c["role"], "ckpt_step": c["ckpt_step"],
                "stage": job["stage"], "eval_seed": cell_seed,
                "cell_type": t, "cell_level": lvl, "episode_idx": ep,
                "disturbance_fired": int(bool(res["disturbance_fired"])),
                "magnitude": res["disturbance_magnitude"], "crash": int(bool(res["is_crash"])),
                "truncation_reason": res["truncation_reason"], "n_steps": len(tr),
                "recovered": None if res["recovered"] is None else int(bool(res["recovered"])),
                "recovery_time_steps": res["recovery_time_steps"],
                "final_pos_error": res["final_pos_error"],
                "mean_pos_error_episode": float(np.mean(tr)), "max_pos_error_episode": float(np.max(tr)),
                "peak_tilt_rad": float(np.max(tl)), "total_reward": res["total_reward"],
                "wind_ss_error": res["wind_steady_state_error_mean"],
            })
    env.close()

    shard = Path(job["shard"])
    tmp = shard.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, shard)
    return c["ckpt_id"]


def run_stage(out: Path, proto: dict, stage: str, ckpts: list[dict], workers: int) -> None:
    scfg = proto[stage]
    shard_dir = out / "episodes" / stage
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for c in ckpts:
        shard = shard_dir / f"{c['ckpt_id']}.csv"
        if shard.exists():
            continue
        jobs.append({"ckpt": c, "path": c["path"], "stage": stage, "stage_preset": proto["stage_preset"],
                     "episodes_per_cell": scfg["episodes_per_cell"], "base_seed": scfg["base_seed"],
                     "shard": str(shard)})
    print(f"[{stage}] {len(ckpts)} checkpoints, {len(jobs)} to run "
          f"({len(ckpts) - len(jobs)} already done), {workers} workers, "
          f"{scfg['episodes_per_cell'] * len(CELLS)} episodes each")
    if not jobs:
        return
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_eval_grid_worker, j): j["ckpt"]["ckpt_id"] for j in jobs}
        for fut in as_completed(futs):
            fut.result()  # re-raise worker errors loudly
            done += 1
            print(f"  [{stage}] {done}/{len(jobs)} done ({futs[fut]})  {time.time() - t0:.0f}s elapsed")


# ----------------------------------------------------------------------------
# Selection logic (reads shards only)
# ----------------------------------------------------------------------------
def _shard_stats(path: Path) -> dict:
    n = crashes = 0
    err = 0.0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            crashes += int(r["crash"])
            err += float(r["final_pos_error"])
    return {"n": n, "crash_rate": crashes / n, "mean_final_pos_error": err / n}


def _rank(out: Path, stage: str, ckpts: list[dict]) -> list[dict]:
    rows = []
    for c in ckpts:
        shard = out / "episodes" / stage / f"{c['ckpt_id']}.csv"
        if not shard.exists():
            sys.exit(f"Missing {stage} shard for {c['ckpt_id']}. Run the `{stage}` command first.")
        rows.append({**c, **_shard_stats(shard)})
    rows.sort(key=lambda r: (r["crash_rate"], r["mean_final_pos_error"], -r["ckpt_step"]))
    return rows


def _by_unit(inv: list[dict], role: str) -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = {}
    for r in inv:
        if r["role"] == role:
            d.setdefault(r["unit"], []).append(r)
    return d


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def cmd_screen(args) -> None:
    out = Path(args.out)
    proto = load_protocol(out)
    inv = read_inventory(out)
    run_stage(out, proto, "screen", [r for r in inv if r["role"] == "candidate"], args.workers)


def _top_k(out: Path, proto: dict, inv: list[dict]) -> list[dict]:
    kept, ranking = [], []
    for unit, cands in sorted(_by_unit(inv, "candidate").items()):
        k = proto["screen"]["top_k"][cands[0]["group"]]
        ranked = _rank(out, "screen", cands)
        for i, r in enumerate(ranked, 1):
            ranking.append({"unit": unit, "group": r["group"], "screen_rank": i, "ckpt_id": r["ckpt_id"],
                            "line": r["line"], "ckpt_step": r["ckpt_step"],
                            "screen_crash_rate": r["crash_rate"],
                            "screen_mean_final_pos_error": r["mean_final_pos_error"], "kept": int(i <= k)})
        kept.extend(ranked[:k])
    (out / "summaries").mkdir(exist_ok=True)
    _write_csv(out / "summaries" / "screen_ranking.csv", ranking)
    return kept


def cmd_select(args) -> None:
    out = Path(args.out)
    proto = load_protocol(out)
    inv = read_inventory(out)
    kept = _top_k(out, proto, inv)
    _write_csv(out / "summaries" / "select_candidates.csv",
               [{"unit": r["unit"], "ckpt_id": r["ckpt_id"], "line": r["line"], "ckpt_step": r["ckpt_step"]}
                for r in kept])
    run_stage(out, proto, "select", kept, args.workers)


def cmd_choose(args) -> None:
    out = Path(args.out)
    proto = load_protocol(out)
    inv = read_inventory(out)
    target = out / "summaries" / "champions.csv"
    if target.exists() and not args.force:
        sys.exit(f"{target} already exists: champions are frozen. Use --force only to deliberately redo.")
    kept = _top_k(out, proto, inv)
    finals_by_unit: dict[str, list[dict]] = {}
    for r in inv:
        if r["role"] == "final":
            finals_by_unit.setdefault(r["unit"], []).append(r)
    champions = []
    for unit in sorted({r["unit"] for r in kept}):
        ranked = _rank(out, "select", [r for r in kept if r["unit"] == unit])
        best = ranked[0]
        row = {"unit": unit, "group": best["group"], "champion_id": best["ckpt_id"],
               "champion_line": best["line"], "champion_step": best["ckpt_step"],
               "champion_sha256": best["sha256"], "select_crash_rate": best["crash_rate"],
               "select_mean_final_pos_error": best["mean_final_pos_error"],
               "n_select_candidates": len(ranked),
               "final_ids": ";".join(f["ckpt_id"] for f in finals_by_unit.get(unit, []))}
        if len(ranked) > 1:
            row["runner_up_id"] = ranked[1]["ckpt_id"]
            row["runner_up_select_crash_rate"] = ranked[1]["crash_rate"]
        champions.append(row)
        print(f"{unit}: champion {best['ckpt_id']} (select crash {best['crash_rate']:.3f})")
    _write_csv(target, champions)
    print(f"Wrote {target}. Champions are now frozen; you may run `report`.")


def cmd_report(args) -> None:
    out = Path(args.out)
    proto = load_protocol(out)
    inv = read_inventory(out)
    champs_path = out / "summaries" / "champions.csv"
    if not champs_path.exists():
        sys.exit("Run `choose` first: champions must be frozen before the held-out report stage.")
    if proto["report"]["scope"] == "all":
        ckpts = inv
    else:
        with open(champs_path, newline="") as f:
            champ_ids = {r["champion_id"] for r in csv.DictReader(f)}
        ckpts = [r for r in inv if r["ckpt_id"] in champ_ids or r["role"] == "final"]
    run_stage(out, proto, "report", ckpts, args.workers)


# ----------------------------------------------------------------------------
# Combine
# ----------------------------------------------------------------------------
def _wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmd_combine(args) -> None:
    import pandas as pd

    out = Path(args.out)
    inv = pd.read_csv(out / "inventory.csv")
    meta = inv[["ckpt_id", "sha256", "file_step"]]
    ckpt_frames, cell_frames = [], []
    for stage in ("screen", "select", "report"):
        shards = sorted((out / "episodes" / stage).glob("*.csv"))
        if not shards:
            continue
        df = pd.concat((pd.read_csv(s) for s in shards), ignore_index=True)
        df.to_csv(out / "episodes" / f"{stage}.csv", index=False)
        keys = ["ckpt_id", "group", "unit", "line", "training_seed", "role", "ckpt_step"]
        s = (df.groupby(keys)
               .agg(n_episodes=("crash", "size"), n_crash=("crash", "sum"), crash_rate=("crash", "mean"),
                    mean_final_pos_error=("final_pos_error", "mean"),
                    mean_pos_error_episode=("mean_pos_error_episode", "mean"))
               .reset_index())
        lo, hi = _wilson(s["n_crash"].to_numpy(float), s["n_episodes"].to_numpy(float))
        s["crash_ci_lo"], s["crash_ci_hi"] = lo, hi
        by_type = (df.pivot_table(index="ckpt_id", columns="cell_type", values="crash", aggfunc="mean")
                     .add_prefix("crash_rate_").reset_index())
        s = s.merge(by_type, on="ckpt_id", how="left")
        s.insert(0, "stage", stage)
        ckpt_frames.append(s.merge(meta, on="ckpt_id", how="left"))

        ck = ["ckpt_id", "group", "unit", "line", "role", "ckpt_step", "cell_type", "cell_level"]
        c = (df.groupby(ck)
               .agg(n=("crash", "size"), n_crash=("crash", "sum"), crash_rate=("crash", "mean"),
                    recovery_rate=("recovered", "mean"), mean_final_pos_error=("final_pos_error", "mean"),
                    peak_tilt_rad=("peak_tilt_rad", "mean")).reset_index())
        lo, hi = _wilson(c["n_crash"].to_numpy(float), c["n"].to_numpy(float))
        c["crash_ci_lo"], c["crash_ci_hi"] = lo, hi
        c.insert(0, "stage", stage)
        cell_frames.append(c)
        print(f"[{stage}] {len(shards)} shards, {len(df)} episodes")
    if not ckpt_frames:
        sys.exit("No shards found. Run screen/select/report first.")
    sm = out / "summaries"
    sm.mkdir(exist_ok=True)
    ck_all = pd.concat(ckpt_frames, ignore_index=True)
    ck_all.to_csv(sm / "checkpoint_summary.csv", index=False)
    pd.concat(cell_frames, ignore_index=True).to_csv(sm / "cell_summary.csv", index=False)

    champs = sm / "champions.csv"
    rep = ck_all[ck_all["stage"] == "report"]
    if champs.exists() and not rep.empty:
        ch = pd.read_csv(champs)
        rows = []
        for _, r in ch.iterrows():
            ids = [("champion", r["champion_id"])]
            if isinstance(r.get("final_ids"), str) and r["final_ids"]:
                ids += [("final", i) for i in r["final_ids"].split(";")]
            for arm, cid in ids:
                m = rep[rep["ckpt_id"] == cid]
                if m.empty:
                    continue
                m = m.iloc[0]
                rows.append({"unit": r["unit"], "group": r["group"], "arm": arm, "ckpt_id": cid,
                             "line": m["line"], "ckpt_step": m["ckpt_step"],
                             "report_n": m["n_episodes"], "report_crash_rate": m["crash_rate"],
                             "report_ci_lo": m["crash_ci_lo"], "report_ci_hi": m["crash_ci_hi"],
                             "report_mean_final_pos_error": m["mean_final_pos_error"]})
        if rows:
            pd.DataFrame(rows).to_csv(sm / "champion_report.csv", index=False)
    print(f"Wrote summaries to {sm}")


# ----------------------------------------------------------------------------
def main() -> None:
    default_workers = max(1, min(12, (os.cpu_count() or 4) // 2))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="freeze protocol and build the checkpoint inventory")
    pi.add_argument("--out", default="results/evaluation/step2")
    pi.add_argument("--chain-seed", type=int, default=0)
    pi.add_argument("--chain-line", default=r"disturbance_3x5(?:_tiltfix\d*)?",
                    help="regex for the training tags that make up the cumulative line")
    pi.add_argument("--no-chain", action="store_true")
    pi.add_argument("--seedstudy-tag", default="seedstudy")
    pi.add_argument("--seedstudy-seeds", default="0-9", help='e.g. "0-9" or "0,1,2"')
    pi.add_argument("--no-seedstudy", action="store_true")
    pi.add_argument("--top-k-chain", type=int, default=10)
    pi.add_argument("--top-k-seedstudy", type=int, default=5)
    pi.add_argument("--report-scope", choices=["all", "champions"], default="all",
                    help="all = held-out eval of every checkpoint (performance-vs-step curves); "
                         "champions = only champions + final saves")
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(fn=cmd_init)

    for name, fn in (("screen", cmd_screen), ("select", cmd_select), ("report", cmd_report)):
        sp = sub.add_parser(name)
        sp.add_argument("--out", default="results/evaluation/step2")
        sp.add_argument("--workers", type=int, default=default_workers)
        sp.set_defaults(fn=fn)

    pc = sub.add_parser("choose", help="pick and freeze per-unit champions from select data")
    pc.add_argument("--out", default="results/evaluation/step2")
    pc.add_argument("--force", action="store_true")
    pc.set_defaults(fn=cmd_choose)

    pm = sub.add_parser("combine", help="merge shards into tidy tables and summaries")
    pm.add_argument("--out", default="results/evaluation/step2")
    pm.set_defaults(fn=cmd_combine)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

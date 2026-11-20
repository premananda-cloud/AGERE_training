# Step-two checkpoint evaluation

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

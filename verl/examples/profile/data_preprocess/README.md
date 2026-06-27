# Profile data preprocessing

Scripts here prepare **parquet datasets for vision-token reuse profiling** only. They live under `examples/profile/` (not upstream `examples/data_preprocess/`) because they depend on profile-specific agents, refocus tools, and experiment layouts.

## Layout

| Path | Used by | Purpose |
|------|---------|---------|
| `chart/download_refocus_chart.py` | `workloads/chart/prepare_data.sh` | HF → multiturn parquet (train/test caps) |
| `chart/filter_refocus_chart_oracle.py` | manual / cleaned rollout | Valid turn0→turn1 gate (oracle, bbox, exec, pixel diff) |
| `chart/refocus_chart_singleturn.py` | Geo3K smoke compare | Multiturn → single-turn for baseline |
| `chart/refocus_chart_dummy_crop.py` | Geo3K smoke compare | Synthetic 2-image crop smoke set |
| `geo3k/geo3k_text_only.py` | `workloads/geo3k/run_geo3k_text_only_profile.sh` | Text-only ablation parquet |
| `deepeyes/download_visual_toolbox_v2.py` | `workloads/deepeyes/prepare_data.sh` | HF DeepEyes 47k → visual_toolbox_v2 parquet |

Upstream Geo3K training data still uses `examples/data_preprocess/geo3k.py`.

## Common commands

```bash
cd verl_vision

# Chart smoke (512 train / 128 test)
bash examples/profile/workloads/chart/prepare_data.sh

# Clean train for oracle refocus profiling (default: train only)
python3 examples/profile/data_preprocess/chart/filter_refocus_chart_oracle.py \
  --input-dir /data/refocus_chart_multiturn \
  --output-dir /data/refocus_chart_multiturn_oracle_changed \
  --progress-every 500

# Outputs per split:
#   train.parquet, train_filter_report.csv, train_filter_summary.json
#
# drop_reason: ok | missing_oracle_code | no_tool_call | unresolved_bbox |
#              exec_failed | no_output_image | unchanged | bad_image | missing_image

# Geo3K text-only ablation
python3 examples/profile/data_preprocess/geo3k/geo3k_text_only.py \
  --match-generate-log profile_logs_geo3k_full/verl_sglang_generate_log_geo3k_full_bs64_n4.csv
```

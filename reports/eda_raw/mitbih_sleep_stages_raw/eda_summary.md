# EDA Summary - mitbih_sleep_stages_raw

## Dataset overview

- Rows: 8645
- Columns: 38
- Numeric columns: 35
- Categorical columns: 3
- Total missing values: 164162
- Target column: `sleep_stage`

## Missing values (top 5 columns)

- `resp_abdomen_mean`: 8206 (94.9219%)
- `resp_abdomen_std`: 8206 (94.9219%)
- `resp_sum_mean`: 8046 (93.0711%)
- `resp_sum_std`: 8046 (93.0711%)
- `eeg_o2_a1_mean`: 8024 (92.8167%)

## Target balance

- `2`: 37.83%
- `W`: 32.01%
- `1`: 17.71%
- `R`: 6.12%
- `3`: 4.85%
- `4`: 1.49%

### Shape indicators (numeric)

Variables with strongest asymmetry (absolute skewness):
- `bp_std`: skewness=9.083, kurtosis=148.730
- `so2_std`: skewness=7.978, kurtosis=82.956
- `so2_mean`: skewness=-7.933, kurtosis=69.522

## Generated figures

- `fig_hist_epoch_end_sample.png`
- `fig_box_epoch_end_sample.png`
- `fig_hist_epoch_start_sample.png`
- `fig_box_epoch_start_sample.png`
- `fig_hist_epoch_end_sec.png`
- `fig_box_epoch_end_sec.png`
- `fig_hist_epoch_start_sec.png`
- `fig_box_epoch_start_sec.png`
- `fig_hist_epoch_index.png`
- `fig_box_epoch_index.png`
- `fig_hist_sv_mean.png`
- `fig_box_sv_mean.png`
- `fig_hist_sv_std.png`
- `fig_box_sv_std.png`
- `fig_hist_bp_mean.png`
- `fig_box_bp_mean.png`
- `fig_hist_so2_mean.png`
- `fig_box_so2_mean.png`
- `fig_hist_bp_std.png`
- `fig_box_bp_std.png`
- `fig_hist_so2_std.png`
- `fig_box_so2_std.png`
- `fig_hist_resp_chest_mean.png`
- `fig_box_resp_chest_mean.png`
- `fig_hist_ecg_mean.png`
- `fig_box_ecg_mean.png`
- `fig_hist_resp_nasal_std.png`
- `fig_box_resp_nasal_std.png`
- `fig_hist_resp_nasal_mean.png`
- `fig_box_resp_nasal_mean.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`

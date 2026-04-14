# EDA Summary - sleep_edf_expanded

## Dataset overview

- Rows: 457652
- Columns: 32
- Numeric columns: 26
- Categorical columns: 6
- Total missing values: 0
- Target column: `sleep_stage`

## Missing values (top 5 columns)

- `eeg_bandpower_alpha`: 0 (0.0%)
- `eeg_bandpower_beta`: 0 (0.0%)
- `eeg_bandpower_delta`: 0 (0.0%)
- `eeg_bandpower_theta`: 0 (0.0%)
- `eeg_channel`: 0 (0.0%)

## Target balance

- `w`: 63.34%
- `n2`: 19.44%
- `rem`: 7.47%
- `n1`: 5.50%
- `n3`: 4.25%

### Shape indicators (numeric)

Variables with strongest asymmetry (absolute skewness):
- `eeg_var`: skewness=170.907, kurtosis=33318.590
- `eeg_rms`: skewness=53.086, kurtosis=5377.399
- `eeg_bandpower_alpha`: skewness=52.602, kurtosis=5951.266

## Generated figures

- `fig_hist_epoch_start_sample.png`
- `fig_box_epoch_start_sample.png`
- `fig_hist_epoch_end_sample.png`
- `fig_box_epoch_end_sample.png`
- `fig_hist_epoch_end_sec.png`
- `fig_box_epoch_end_sec.png`
- `fig_hist_epoch_start_sec.png`
- `fig_box_epoch_start_sec.png`
- `fig_hist_epoch_index.png`
- `fig_box_epoch_index.png`
- `fig_hist_eeg_zero_crossing_rate.png`
- `fig_box_eeg_zero_crossing_rate.png`
- `fig_hist_eeg_theta_alpha_ratio.png`
- `fig_box_eeg_theta_alpha_ratio.png`
- `fig_hist_sleep_stage_w.png`
- `fig_box_sleep_stage_w.png`
- `fig_hist_eeg_rel_power_alpha.png`
- `fig_box_eeg_rel_power_alpha.png`
- `fig_hist_eeg_bandpower_alpha.png`
- `fig_box_eeg_bandpower_alpha.png`
- `fig_hist_sleep_stage_rem.png`
- `fig_box_sleep_stage_rem.png`
- `fig_hist_sleep_stage_n2.png`
- `fig_box_sleep_stage_n2.png`
- `fig_hist_eeg_rel_power_delta.png`
- `fig_box_eeg_rel_power_delta.png`
- `fig_hist_eeg_bandpower_delta.png`
- `fig_box_eeg_bandpower_delta.png`
- `fig_hist_eeg_std.png`
- `fig_box_eeg_std.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`

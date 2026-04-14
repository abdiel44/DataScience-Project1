# EDA Summary - isruc_sleep

## Dataset overview

- Rows: 22320
- Columns: 25
- Numeric columns: 20
- Categorical columns: 5
- Total missing values: 0
- Target column: `sleep_stage`

## Missing values (top 5 columns)

- `eeg_bandpower_alpha`: 0 (0.0%)
- `eeg_bandpower_beta`: 0 (0.0%)
- `eeg_bandpower_delta`: 0 (0.0%)
- `eeg_bandpower_theta`: 0 (0.0%)
- `eeg_channel`: 0 (0.0%)

## Target balance

- `3`: 48.85%
- `2`: 38.28%
- `1`: 12.87%

### Shape indicators (numeric)

Variables with strongest asymmetry (absolute skewness):
- `eeg_bandpower_delta`: skewness=29.821, kurtosis=1089.714
- `eeg_bandpower_theta`: skewness=21.981, kurtosis=661.805
- `eeg_bandpower_alpha`: skewness=21.860, kurtosis=1276.636

## Generated figures

- `fig_hist_eeg_bandpower_beta.png`
- `fig_box_eeg_bandpower_beta.png`
- `fig_hist_eeg_std.png`
- `fig_box_eeg_std.png`
- `fig_hist_eeg_var.png`
- `fig_box_eeg_var.png`
- `fig_hist_eeg_theta_alpha_ratio.png`
- `fig_box_eeg_theta_alpha_ratio.png`
- `fig_hist_eeg_spectral_entropy.png`
- `fig_box_eeg_spectral_entropy.png`
- `fig_hist_sleep_stage_1.png`
- `fig_box_sleep_stage_1.png`
- `fig_hist_eeg_mean.png`
- `fig_box_eeg_mean.png`
- `fig_hist_eeg_rms.png`
- `fig_box_eeg_rms.png`
- `fig_hist_eeg_bandpower_alpha.png`
- `fig_box_eeg_bandpower_alpha.png`
- `fig_hist_eeg_rel_power_theta.png`
- `fig_box_eeg_rel_power_theta.png`
- `fig_hist_eeg_bandpower_delta.png`
- `fig_box_eeg_bandpower_delta.png`
- `fig_hist_eeg_rel_power_delta.png`
- `fig_box_eeg_rel_power_delta.png`
- `fig_hist_sleep_stage_3.png`
- `fig_box_sleep_stage_3.png`
- `fig_hist_eeg_rel_power_beta.png`
- `fig_box_eeg_rel_power_beta.png`
- `fig_hist_eeg_rel_power_alpha.png`
- `fig_box_eeg_rel_power_alpha.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`

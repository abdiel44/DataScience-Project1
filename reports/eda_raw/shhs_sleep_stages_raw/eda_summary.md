# EDA Summary - shhs_sleep_stages_raw

## Dataset overview

- Rows: 1080
- Columns: 29
- Numeric columns: 28
- Categorical columns: 1
- Total missing values: 0
- Target column: `sleep_stage`

## Missing values (top 5 columns)

- `abdo_res_mean`: 0 (0.0%)
- `abdo_res_std`: 0 (0.0%)
- `airflow_mean`: 0 (0.0%)
- `airflow_std`: 0 (0.0%)
- `ecg_mean`: 0 (0.0%)

## Target balance

- `2`: 37.78%
- `3`: 22.04%
- `W`: 19.91%
- `R`: 18.15%
- `1`: 2.13%

### Shape indicators (numeric)

Variables with strongest asymmetry (absolute skewness):
- `sao2_std`: skewness=13.656, kurtosis=203.258
- `sao2_mean`: skewness=-9.996, kurtosis=191.761
- `emg_mean`: skewness=-7.458, kurtosis=70.867

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
- `fig_hist_eog_l_std.png`
- `fig_box_eog_l_std.png`
- `fig_hist_eog_r_std.png`
- `fig_box_eog_r_std.png`
- `fig_hist_pr_mean.png`
- `fig_box_pr_mean.png`
- `fig_hist_eeg_sec_std.png`
- `fig_box_eeg_sec_std.png`
- `fig_hist_eeg_std.png`
- `fig_box_eeg_std.png`
- `fig_hist_sao2_std.png`
- `fig_box_sao2_std.png`
- `fig_hist_pr_std.png`
- `fig_box_pr_std.png`
- `fig_hist_sao2_mean.png`
- `fig_box_sao2_mean.png`
- `fig_hist_airflow_std.png`
- `fig_box_airflow_std.png`
- `fig_hist_eog_l_mean.png`
- `fig_box_eog_l_mean.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`

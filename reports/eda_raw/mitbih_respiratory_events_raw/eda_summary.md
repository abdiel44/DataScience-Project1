# EDA Summary - mitbih_respiratory_events_raw

## Dataset overview

- Rows: 3781
- Columns: 39
- Numeric columns: 35
- Categorical columns: 4
- Total missing values: 70454
- Target column: `event_tokens`

## Missing values (top 5 columns)

- `eog_mean`: 3651 (96.5618%)
- `eog_std`: 3651 (96.5618%)
- `eeg_o2_a1_mean`: 3582 (94.7368%)
- `eeg_o2_a1_std`: 3582 (94.7368%)
- `resp_abdomen_mean`: 3552 (93.9434%)

## Target balance

- `X`: 38.19%
- `HA`: 18.22%
- `OA`: 15.26%
- `H`: 5.92%
- `L`: 5.53%
- `CAA`: 4.05%
- `LA`: 3.76%
- `X X`: 1.75%
- `CA`: 1.38%
- `LA LA`: 0.71%
- `A`: 0.63%
- `HA HA`: 0.42%
- `OA OA`: 0.42%
- `L L`: 0.42%
- `L LA`: 0.32%
- `HA X`: 0.24%
- `X L`: 0.24%
- `H H`: 0.21%
- `HA H`: 0.16%
- `X H`: 0.13%
- `CAA CAA`: 0.13%
- `H LA`: 0.13%
- `CAA L`: 0.13%
- `LA L`: 0.11%
- `CAA HA`: 0.11%
- `L A`: 0.11%
- `X OA`: 0.08%
- `HA CAA`: 0.08%
- `L H`: 0.08%
- `H HA`: 0.08%
- `L X`: 0.08%
- `H X`: 0.08%
- `H L`: 0.08%
- `H CA`: 0.05%
- `HA L`: 0.05%
- `LA HA`: 0.05%
- `L L L`: 0.05%
- `OA L`: 0.05%
- `LA H`: 0.05%
- `X CAA`: 0.05%
- `L HA LA`: 0.03%
- `L HA`: 0.03%
- `LA HA LA`: 0.03%
- `A A A`: 0.03%
- `X A`: 0.03%
- `X CA`: 0.03%
- `A X`: 0.03%
- `LA X`: 0.03%
- `L OA`: 0.03%
- `CAA X`: 0.03%
- `HA CA`: 0.03%
- `L CAA`: 0.03%
- `OA X`: 0.03%
- `X LA`: 0.03%
- `CAA CAA L`: 0.03%

### Shape indicators (numeric)

Variables with strongest asymmetry (absolute skewness):
- `eeg_o2_a1_mean`: skewness=8.671, kurtosis=101.780
- `so2_mean`: skewness=-8.360, kurtosis=81.008
- `resp_nasal_std`: skewness=8.072, kurtosis=92.313

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
- `fig_hist_ecg_mean.png`
- `fig_box_ecg_mean.png`
- `fig_hist_resp_chest_mean.png`
- `fig_box_resp_chest_mean.png`
- `fig_hist_resp_nasal_std.png`
- `fig_box_resp_nasal_std.png`
- `fig_hist_resp_nasal_mean.png`
- `fig_box_resp_nasal_mean.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`

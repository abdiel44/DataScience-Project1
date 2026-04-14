# EDA Summary - sleep_edf_expanded_raw

## Dataset overview

- Rows: 197
- Columns: 16
- Numeric columns: 11
- Categorical columns: 5
- Total missing values: 0
- Target column: `sleep_stage`

## Missing values (top 5 columns)

- `duration_sec`: 0 (0.0%)
- `hypnogram_file`: 0 (0.0%)
- `known_stage_duration_sec`: 0 (0.0%)
- `n_channels`: 0 (0.0%)
- `n_hypnogram_annotations`: 0 (0.0%)

## Target balance

- `W`: 78.17%
- `N2`: 20.81%
- `N3`: 0.51%
- `REM`: 0.51%

### Shape indicators (numeric)

Variables with strongest asymmetry (absolute skewness):
- `unknown_stage_duration_sec`: skewness=2.396, kurtosis=8.430
- `sleep_stage_frac_n3`: skewness=1.713, kurtosis=2.543
- `sleep_stage_frac_rem`: skewness=1.639, kurtosis=2.142

## Generated figures

- `fig_hist_known_stage_duration_sec.png`
- `fig_box_known_stage_duration_sec.png`
- `fig_hist_duration_sec.png`
- `fig_box_duration_sec.png`
- `fig_hist_unknown_stage_duration_sec.png`
- `fig_box_unknown_stage_duration_sec.png`
- `fig_hist_n_hypnogram_annotations.png`
- `fig_box_n_hypnogram_annotations.png`
- `fig_hist_n_channels.png`
- `fig_box_n_channels.png`
- `fig_hist_sleep_stage_frac_w.png`
- `fig_box_sleep_stage_frac_w.png`
- `fig_hist_sleep_stage_frac_n2.png`
- `fig_box_sleep_stage_frac_n2.png`
- `fig_hist_sleep_stage_frac_n3.png`
- `fig_box_sleep_stage_frac_n3.png`
- `fig_hist_sleep_stage_frac_rem.png`
- `fig_box_sleep_stage_frac_rem.png`
- `fig_hist_sleep_stage_frac_n1.png`
- `fig_box_sleep_stage_frac_n1.png`
- `fig_hist_sfreq_first.png`
- `fig_box_sfreq_first.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`

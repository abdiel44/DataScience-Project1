# EDA Summary - st_vincent_apnea

## Dataset overview

- Rows: 25
- Columns: 10
- Numeric columns: 9
- Categorical columns: 1
- Total missing values: 0
- Target column: `stage_mode`

## Missing values (top 5 columns)

- `n_epochs`: 0 (0.0%)
- `recording_id`: 0 (0.0%)
- `stage_0_frac`: 0 (0.0%)
- `stage_1_frac`: 0 (0.0%)
- `stage_2_frac`: 0 (0.0%)

## Target balance

- `3`: 48.00%
- `0`: 28.00%
- `2`: 16.00%
- `5`: 4.00%
- `1`: 4.00%

### Shape indicators (numeric)

Variables with strongest asymmetry (absolute skewness):
- `stage_2_frac`: skewness=1.470, kurtosis=2.603
- `stage_5_frac`: skewness=0.492, kurtosis=0.230
- `stage_0_frac`: skewness=0.476, kurtosis=-1.165

## Generated figures

- `fig_hist_stage_mode.png`
- `fig_box_stage_mode.png`
- `fig_hist_stage_2_frac.png`
- `fig_box_stage_2_frac.png`
- `fig_hist_stage_1_frac.png`
- `fig_box_stage_1_frac.png`
- `fig_hist_stage_0_frac.png`
- `fig_box_stage_0_frac.png`
- `fig_hist_stage_median.png`
- `fig_box_stage_median.png`
- `fig_hist_stage_5_frac.png`
- `fig_box_stage_5_frac.png`
- `fig_hist_n_epochs.png`
- `fig_box_n_epochs.png`
- `fig_hist_stage_3_frac.png`
- `fig_box_stage_3_frac.png`
- `fig_hist_stage_4_frac.png`
- `fig_box_stage_4_frac.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`

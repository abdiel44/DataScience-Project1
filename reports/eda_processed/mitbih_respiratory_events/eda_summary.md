# EDA Summary - mitbih_respiratory_events

## Dataset overview

- Rows: 3781
- Columns: 94
- Numeric columns: 90
- Categorical columns: 4
- Total missing values: 0
- Target column: `event_tokens`

## Missing values (top 5 columns)

- `aux_raw`: 0 (0.0%)
- `bp_mean`: 0 (0.0%)
- `bp_std`: 0 (0.0%)
- `ecg_mean`: 0 (0.0%)
- `ecg_std`: 0 (0.0%)

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
- `event_tokens_OA X`: skewness=61.490, kurtosis=3781.000
- `event_tokens_CAA CAA L`: skewness=61.490, kurtosis=3781.000
- `event_tokens_CAA X`: skewness=61.490, kurtosis=3781.000

## Generated figures

- `fig_hist_event_tokens_HA_CA.png`
- `fig_box_event_tokens_HA_CA.png`
- `fig_hist_event_tokens_CAA_CAA_L.png`
- `fig_box_event_tokens_CAA_CAA_L.png`
- `fig_hist_event_tokens_L_CAA.png`
- `fig_box_event_tokens_L_CAA.png`
- `fig_hist_event_tokens_LA_X.png`
- `fig_box_event_tokens_LA_X.png`
- `fig_hist_event_tokens_X_CA.png`
- `fig_box_event_tokens_X_CA.png`
- `fig_hist_event_tokens_H_H.png`
- `fig_box_event_tokens_H_H.png`
- `fig_hist_event_tokens_HA_HA.png`
- `fig_box_event_tokens_HA_HA.png`
- `fig_hist_event_tokens_CAA_L.png`
- `fig_box_event_tokens_CAA_L.png`
- `fig_hist_event_tokens_L_LA.png`
- `fig_box_event_tokens_L_LA.png`
- `fig_hist_event_tokens_L.png`
- `fig_box_event_tokens_L.png`
- `fig_hist_event_tokens_X_L.png`
- `fig_box_event_tokens_X_L.png`
- `fig_hist_event_tokens_LA_LA.png`
- `fig_box_event_tokens_LA_LA.png`
- `fig_hist_event_tokens_CAA_HA.png`
- `fig_box_event_tokens_CAA_HA.png`
- `fig_hist_event_tokens_L_L.png`
- `fig_box_event_tokens_L_L.png`
- `fig_hist_resp_abdominal_std.png`
- `fig_box_resp_abdominal_std.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`

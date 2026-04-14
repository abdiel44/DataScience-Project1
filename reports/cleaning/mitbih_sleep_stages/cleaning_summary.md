# Data cleaning summary - mitbih_sleep_stages

## Row and column changes

- Input rows: 8645
- Output rows: 8645
- Removed fully empty rows: 0
- Removed duplicate rows (full row duplicate): 0
- Removed rows with missing target: 0

## Columns dropped (high missing %)

- None

## Missing values (top 10 columns, before imputation)

- `resp_abdomen_mean`: 8206
- `resp_abdomen_std`: 8206
- `resp_sum_mean`: 8046
- `resp_sum_std`: 8046
- `eeg_o2_a1_mean`: 8024
- `eeg_o2_a1_std`: 8024
- `eog_mean`: 8005
- `eog_std`: 8005
- `eog_right_mean`: 7188
- `eog_right_std`: 7188

## Numeric coercion (object to numeric)

- None

## String normalization (strip, collapse spaces, lowercase)

- `record_id`
- `sleep_stage`

## Outliers (none)

- No winsorization applied or no numeric columns processed.

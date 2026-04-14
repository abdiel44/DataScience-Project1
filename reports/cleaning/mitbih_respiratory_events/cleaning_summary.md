# Data cleaning summary - mitbih_respiratory_events

## Row and column changes

- Input rows: 3781
- Output rows: 3781
- Removed fully empty rows: 0
- Removed duplicate rows (full row duplicate): 0
- Removed rows with missing target: 0

## Columns dropped (high missing %)

- None

## Missing values (top 10 columns, before imputation)

- `eog_mean`: 3651
- `eog_std`: 3651
- `eeg_o2_a1_mean`: 3582
- `eeg_o2_a1_std`: 3582
- `resp_abdomen_mean`: 3552
- `resp_abdomen_std`: 3552
- `resp_sum_mean`: 3502
- `resp_sum_std`: 3502
- `resp_chest_mean`: 3284
- `resp_chest_std`: 3284

## Numeric coercion (object to numeric)

- None

## String normalization (strip, collapse spaces, lowercase)

- `record_id`
- `sleep_stage`

## Outliers (none)

- No winsorization applied or no numeric columns processed.

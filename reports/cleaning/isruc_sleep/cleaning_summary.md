# Data cleaning summary - isruc_sleep

## Row and column changes

- Input rows: 22320
- Output rows: 22320
- Removed fully empty rows: 0
- Removed duplicate rows (full row duplicate): 0
- Removed rows with missing target: 0

## Columns dropped (high missing %)

- None

## Missing values (top 10 columns, before imputation)

- `source_file`: 0
- `subject_unit_id`: 0
- `event_group`: 0
- `sleep_stage`: 0
- `eeg_channel`: 0
- `sfreq_hz`: 0
- `epoch_duration_sec`: 0
- `eeg_mean`: 0
- `eeg_std`: 0
- `eeg_var`: 0

## Numeric coercion (object to numeric)

- None

## String normalization (strip, collapse spaces, lowercase)

- `event_group`
- `eeg_channel`

## Outliers (none)

- No winsorization applied or no numeric columns processed.

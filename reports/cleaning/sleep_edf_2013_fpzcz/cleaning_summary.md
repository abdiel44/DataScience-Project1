# Data cleaning summary - sleep_edf_2013_fpzcz

## Row and column changes

- Input rows: 195479
- Output rows: 195479
- Removed fully empty rows: 0
- Removed duplicate rows (full row duplicate): 0
- Removed rows with missing target: 0

## Columns dropped (high missing %)

- None

## Missing values (top 10 columns, before imputation)

- `recording_id`: 0
- `subject_id`: 0
- `epoch_index`: 0
- `epoch_start_sample`: 0
- `epoch_end_sample`: 0
- `epoch_start_sec`: 0
- `epoch_end_sec`: 0
- `sleep_stage`: 0
- `eeg_channel`: 0
- `sfreq_hz`: 0

## Numeric coercion (object to numeric)

- None

## String normalization (strip, collapse spaces, lowercase)

- `sleep_stage`
- `eeg_channel`
- `sleep_edf_variant`
- `sleep_edf_subset`

## Outliers (none)

- No winsorization applied or no numeric columns processed.

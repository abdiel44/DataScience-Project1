# Variable encoding summary

## Topic alignment

- **Nominal** columns (no natural order): one-hot encoding → `added_dummy_columns`.
- **Ordinal** columns: integer codes 0..n-1 following the declared order.
- **Binary** columns: mapped to 0.0 / 1.0.
- Numeric scaling: use `scaling.py` / `--scale-method` after encoding.

## Applied rules

- Nominal: `['sleep_stage']`
- Ordinal: `[]`
- Binary: `[]`
- `drop_first_dummy`: False

## New dummy columns

- `sleep_stage_1`
- `sleep_stage_2`
- `sleep_stage_3`
- `sleep_stage_r`
- `sleep_stage_w`

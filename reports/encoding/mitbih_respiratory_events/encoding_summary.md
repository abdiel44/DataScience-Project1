# Variable encoding summary

## Topic alignment

- **Nominal** columns (no natural order): one-hot encoding → `added_dummy_columns`.
- **Ordinal** columns: integer codes 0..n-1 following the declared order.
- **Binary** columns: mapped to 0.0 / 1.0.
- Numeric scaling: use `scaling.py` / `--scale-method` after encoding.

## Applied rules

- Nominal: `['event_tokens']`
- Ordinal: `[]`
- Binary: `[]`
- `drop_first_dummy`: False

## New dummy columns

- `event_tokens_A`
- `event_tokens_A A A`
- `event_tokens_A X`
- `event_tokens_CA`
- `event_tokens_CAA`
- `event_tokens_CAA CAA`
- `event_tokens_CAA CAA L`
- `event_tokens_CAA HA`
- `event_tokens_CAA L`
- `event_tokens_CAA X`
- `event_tokens_H`
- `event_tokens_H CA`
- `event_tokens_H H`
- `event_tokens_H HA`
- `event_tokens_H L`
- `event_tokens_H LA`
- `event_tokens_H X`
- `event_tokens_HA`
- `event_tokens_HA CA`
- `event_tokens_HA CAA`
- `event_tokens_HA H`
- `event_tokens_HA HA`
- `event_tokens_HA L`
- `event_tokens_HA X`
- `event_tokens_L`
- `event_tokens_L A`
- `event_tokens_L CAA`
- `event_tokens_L H`
- `event_tokens_L HA`
- `event_tokens_L HA LA`
- `event_tokens_L L`
- `event_tokens_L L L`
- `event_tokens_L LA`
- `event_tokens_L OA`
- `event_tokens_L X`
- `event_tokens_LA`
- `event_tokens_LA H`
- `event_tokens_LA HA`
- `event_tokens_LA HA LA`
- `event_tokens_LA L`
- `event_tokens_LA LA`
- `event_tokens_LA X`
- `event_tokens_OA`
- `event_tokens_OA L`
- `event_tokens_OA OA`
- `event_tokens_OA X`
- `event_tokens_X`
- `event_tokens_X A`
- `event_tokens_X CA`
- `event_tokens_X CAA`
- `event_tokens_X H`
- `event_tokens_X L`
- `event_tokens_X LA`
- `event_tokens_X OA`
- `event_tokens_X X`

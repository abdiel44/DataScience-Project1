# Configuraciones congeladas por dataset (`config/datasets/`)

Cumple **C3** del [PRD](PRD.md): especificaciones versionadas para `--encoding-spec` en [`src/main.py`](../src/main.py). Ajusta nombres de columnas en el JSON si tu CSV difiere; el pipeline normaliza a `snake_case`.

| Archivo | Uso típico | `target_column` |
|---------|------------|-----------------|
| [isruc_sleep.json](../config/datasets/isruc_sleep.json) | ISRUC staging por segmento con sujeto desde filename y `eeg_*` genérico | `sleep_stage` |
| [mitbih_sleep_stages.json](../config/datasets/mitbih_sleep_stages.json) | Export WFDB MIT-BIH, tarea estadificación | `sleep_stage` |
| [mitbih_respiratory_events.json](../config/datasets/mitbih_respiratory_events.json) | Export MIT-BIH, eventos en anotaciones | `event_tokens` |
| [shhs_sleep_stages.json](../config/datasets/shhs_sleep_stages.json) | Export SHHS, hipnograma | `sleep_stage` |
| [shhs_respiratory_events.json](../config/datasets/shhs_respiratory_events.json) | Export SHHS, eventos resp/arou | `event_label` |
| [st_vincent_apnea.json](../config/datasets/st_vincent_apnea.json) | Ingesta St. Vincent | `stage_mode` |
| [sleep_edf_expanded.json](../config/datasets/sleep_edf_expanded.json) | Sleep-EDF epoch-level desde PSG/hypnogram EDF con `eeg_*` genérico | `sleep_stage` |

## Comando tipo

```bash
python src/main.py --input data/processed/sleep_edf_expanded_raw.csv --output data/processed/sleep_edf_expanded_prep.csv --encoding-spec config/datasets/sleep_edf_expanded.json --target-col sleep_stage --task sleep_edf_expanded --write-cleaning-report
```

Añade según necesidad: `--run-eda`, `--write-encoding-report`, `--write-scaling-report`, `--balance-method`, `--scale-method`, `--dimensionality-method`. Para Fase E desde YAML, usa los presets de [`config/experiments/`](../config/experiments/README.md).

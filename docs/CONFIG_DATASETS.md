# Configuraciones congeladas por dataset (`config/datasets/`)

Cumple **C3** del [PRD](PRD.md): especificaciones versionadas para `--encoding-spec` en [`src/main.py`](../src/main.py). Ajusta nombres de columnas en el JSON si tu CSV difiere (el pipeline normaliza a **snake_case**).

| Archivo | Uso típico | `target_column` (referencia) |
|---------|------------|------------------------------|
| [isruc_sleep.json](../config/datasets/isruc_sleep.json) | Ingesta ISRUC agregada por segmento | `event_group` (filas incluyen `subject_unit_id`, `source_file`) |
| [mitbih_sleep_stages.json](../config/datasets/mitbih_sleep_stages.json) | Export WFDB MIT-BIH, tarea estadificación | `sleep_stage` |
| [mitbih_respiratory_events.json](../config/datasets/mitbih_respiratory_events.json) | Export MIT-BIH, eventos en anotaciones | `event_tokens` |
| [shhs_sleep_stages.json](../config/datasets/shhs_sleep_stages.json) | Export SHHS, hipnograma | `sleep_stage` |
| [shhs_respiratory_events.json](../config/datasets/shhs_respiratory_events.json) | Export SHHS, eventos resp/arou | `event_label` |
| [st_vincent_apnea.json](../config/datasets/st_vincent_apnea.json) | Ingesta St. Vincent | `stage_mode` |
| [sleep_edf_expanded.json](../config/datasets/sleep_edf_expanded.json) | CSV propio tras EDF/MNE u otra exportación | `sleep_stage` (ajustar si aplica) |

## Comando tipo (post-export o CSV procesado)

```bash
python src/main.py --input data/processed/mitbih_sleep_stages.csv --output data/processed/mitbih_sleep_stages_prep.csv --encoding-spec config/datasets/mitbih_sleep_stages.json --target-col sleep_stage --task mitbih_staging --write-cleaning-report
```

Añade según necesidad: `--run-eda`, `--balance-method`, `--scale-method`, `--dimensionality-method`, informes `--write-encoding-report`, etc. Ver [README.md](../README.md) y [reproducir.txt](../reproducir.txt).

---

*Al cambiar hiperparámetros de preprocesamiento para el informe final, actualizar el JSON correspondiente y el commit git para trazabilidad.*

Para **Fase E** (modelado desde YAML), los presets alineados con estas tablas están en [`config/experiments/`](../config/experiments/README.md); ver también [FASE_E_PLUG_AND_PLAY.md](FASE_E_PLUG_AND_PLAY.md).

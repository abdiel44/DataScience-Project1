# Presets Fase E (plug-and-play)

YAML listos para copiar o usar directamente con [`scripts/run_phase_e_cv.py`](../../scripts/run_phase_e_cv.py), alineados con el [PRD §4](../docs/PRD.md) y los exports/ingestas de este repo.

| Preset | CSV esperado | `subject_column` | `target_column` | `task` |
|--------|--------------|------------------|-----------------|--------|
| [mitbih_sleep_stages.yaml](mitbih_sleep_stages.yaml) | Export/ingesta MIT-BIH épocas sueño | `record_id` | `sleep_stage` | multiclass |
| [mitbih_respiratory_events.yaml](mitbih_respiratory_events.yaml) | CSV eventos MIT-BIH | `record_id` | `event_tokens` | multiclass |
| [shhs_sleep_stages.yaml](shhs_sleep_stages.yaml) | Export/ingesta SHHS hipnograma | `record_id` | `sleep_stage` | multiclass |
| [shhs_respiratory_events.yaml](shhs_respiratory_events.yaml) | CSV eventos SHHS | `record_id` | `event_label` | multiclass |
| [isruc_sleep_event_group.yaml](isruc_sleep_event_group.yaml) | Ingesta ISRUC (`--source isruc-sleep`) | `subject_unit_id` | `event_group` | multiclass |
| [isruc_sleep_sleep_stage.yaml](isruc_sleep_sleep_stage.yaml) | ISRUC (filas con `sleep_stage` numérico) | `subject_unit_id` | `sleep_stage` | multiclass |
| [st_vincent_apnea.yaml](st_vincent_apnea.yaml) | Ingesta St. Vincent | `recording_id` | `stage_mode` | multiclass |
| [sleep_edf_expanded.yaml](sleep_edf_expanded.yaml) | CSV propio post-EDF | `recording_id` | `sleep_stage` | multiclass |
| [cross_dataset_mitbih_shhs_template.yaml](cross_dataset_mitbih_shhs_template.yaml) | Entrenar A / evaluar B (rellenar rutas) | — | — | — |

Guía detallada (PDF del curso, tuning, binario apnea, EEG monocanal): [docs/FASE_E_PLUG_AND_PLAY.md](../../docs/FASE_E_PLUG_AND_PLAY.md).

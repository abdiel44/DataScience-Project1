# Presets Fase E (plug-and-play)

YAML listos para copiar o usar directamente con [`scripts/run_phase_e_cv.py`](../../scripts/run_phase_e_cv.py), alineados con el [PRD](../docs/PRD.md) y los exports/ingestas del repo.

| Preset | CSV esperado | `subject_column` | `target_column` | `task` |
|--------|--------------|------------------|-----------------|--------|
| [mitbih_sleep_stages.yaml](mitbih_sleep_stages.yaml) | Export/ingesta MIT-BIH epocas sueno | `record_id` | `sleep_stage` | multiclass |
| [mitbih_respiratory_events.yaml](mitbih_respiratory_events.yaml) | CSV eventos MIT-BIH | `record_id` | `event_tokens` | multiclass |
| [shhs_sleep_stages.yaml](shhs_sleep_stages.yaml) | Export/ingesta SHHS hipnograma | `record_id` | `sleep_stage` | multiclass |
| [shhs_respiratory_events.yaml](shhs_respiratory_events.yaml) | CSV eventos SHHS | `record_id` | `event_label` | multiclass |
| [isruc_sleep_event_group.yaml](isruc_sleep_event_group.yaml) | ISRUC legacy por `event_group` | `subject_unit_id` | `event_group` | multiclass |
| [isruc_sleep_sleep_stage.yaml](isruc_sleep_sleep_stage.yaml) | ISRUC staging con sujeto real desde filename + `eeg_*` | `subject_unit_id` | `sleep_stage` | multiclass |
| [st_vincent_apnea.yaml](st_vincent_apnea.yaml) | Ingesta St. Vincent | `recording_id` | `stage_mode` | multiclass |
| [sleep_edf_expanded.yaml](sleep_edf_expanded.yaml) | Sleep-EDF epoch-level con `eeg_*` generico | `subject_id` | `sleep_stage` | multiclass |
| [sleep_edf_2013_fpzcz.yaml](sleep_edf_2013_fpzcz.yaml) | Sleep-EDF 2013 comparable a SleepEEGNet (`Fpz-Cz`, wake-trim, contexto temporal) | `subject_id` | `sleep_stage` | multiclass |
| [sleep_edf_2013_fpzcz_tuned.yaml](sleep_edf_2013_fpzcz_tuned.yaml) | Igual que arriba + nested CV y resampling train-only | `subject_id` | `sleep_stage` | multiclass |
| [sleep_edf_2013_fpzcz_deep_cnn.yaml](sleep_edf_2013_fpzcz_deep_cnn.yaml) | Sleep-EDF 2013 comparable, waveform + `CNN` local-first | `subject_id` | `sleep_stage` | multiclass |
| [sleep_edf_2013_fpzcz_deep_conformer.yaml](sleep_edf_2013_fpzcz_deep_conformer.yaml) | Sleep-EDF 2013 comparable, waveform + `Conformer` pequeno | `subject_id` | `sleep_stage` | multiclass |
| [sleep_edf_2013_fpzcz_deep_conformer_ssl.yaml](sleep_edf_2013_fpzcz_deep_conformer_ssl.yaml) | Igual que arriba + pretraining contrastivo ligero | `subject_id` | `sleep_stage` | multiclass |
| [sleep_edf_expanded_multitask_pretrain.yaml](sleep_edf_expanded_multitask_pretrain.yaml) | Pretraining multitarea/base transfer usando Sleep-EDF Expanded con staging auxiliar | `subject_unit_id` | - | multitask |
| [mitbih_apnea_binary_multitask.yaml](mitbih_apnea_binary_multitask.yaml) | MIT-BIH apnea/no-apnea + staging auxiliar con transferencia | `subject_unit_id` | - | multitask |
| [shhs_apnea_binary_multitask.yaml](shhs_apnea_binary_multitask.yaml) | SHHS apnea/no-apnea + staging auxiliar con transferencia | `subject_unit_id` | - | multitask |
| [st_vincent_apnea_binary_multitask.yaml](st_vincent_apnea_binary_multitask.yaml) | St. Vincent apnea/no-apnea + staging auxiliar con transferencia | `subject_unit_id` | - | multitask |
| [cross_dataset_mitbih_to_shhs_apnea_multitask.yaml](cross_dataset_mitbih_to_shhs_apnea_multitask.yaml) | Train MIT-BIH / eval SHHS para cambio de dominio | `subject_unit_id` | - | multitask |
| [mitbih_apnea_stage_classic.yaml](mitbih_apnea_stage_classic.yaml) | Linea clasica multitarget MIT-BIH con `sleep_stage` + `apnea_binary` | `subject_unit_id` | - | classic_multitarget |
| [st_vincent_apnea_stage_classic.yaml](st_vincent_apnea_stage_classic.yaml) | Linea clasica multitarget St. Vincent con `sleep_stage` + `apnea_binary` | `subject_unit_id` | - | classic_multitarget |
| [shhs_apnea_stage_classic.yaml](shhs_apnea_stage_classic.yaml) | SHHS clasico multitarget para holdout externo | `subject_unit_id` | - | classic_multitarget |
| [cross_dataset_mitbih_to_st_vincent_classic.yaml](cross_dataset_mitbih_to_st_vincent_classic.yaml) | Train MIT-BIH / eval St. Vincent con features `eeg_*` compartidas | `subject_unit_id` | - | classic_multitarget |
| [cross_dataset_st_vincent_to_mitbih_classic.yaml](cross_dataset_st_vincent_to_mitbih_classic.yaml) | Train St. Vincent / eval MIT-BIH con features `eeg_*` compartidas | `subject_unit_id` | - | classic_multitarget |
| [cross_dataset_mitbih_to_shhs_classic.yaml](cross_dataset_mitbih_to_shhs_classic.yaml) | Train MIT-BIH / eval SHHS como holdout externo | `subject_unit_id` | - | classic_multitarget |
| [cross_dataset_mitbih_shhs_template.yaml](cross_dataset_mitbih_shhs_template.yaml) | Entrenar A / evaluar B (rellenar rutas) | - | - | - |

Guia detallada: [docs/FASE_E_PLUG_AND_PLAY.md](../../docs/FASE_E_PLUG_AND_PLAY.md).

Linea deep:
- script: [`scripts/run_phase_e_deep.py`](../../scripts/run_phase_e_deep.py)
- esquema: [experiment_train_deep.schema.md](../experiment_train_deep.schema.md)
- con `dataset.input_mode: raw` usa `*_raw.csv` y reconstruye la senal desde `data/raw/`
- con `dataset.input_mode: epoch_store` usa un manifiesto materializado + `.npy` por grabacion; generar antes con [`scripts/materialize_epoch_store.py`](../../scripts/materialize_epoch_store.py)

Linea multitask apnea:
- preparar metadata: [`scripts/prepare_multitask_apnea_metadata.py`](../../scripts/prepare_multitask_apnea_metadata.py)
- runner: [`scripts/run_phase_e_multitask.py`](../../scripts/run_phase_e_multitask.py)
- usa CSVs con `dataset_id`, `apnea_binary`, `sleep_stage` opcional
- para entrenamiento rapido, materializar `epoch_store` con [`scripts/materialize_epoch_store.py`](../../scripts/materialize_epoch_store.py) y correr con `dataset.input_mode: epoch_store`

Linea clasica multitarget:
- exportar tabla EEG canonica desde `epoch_store` con [`scripts/prepare_classic_multitarget_features.py`](../../scripts/prepare_classic_multitarget_features.py)
- runner: [`scripts/run_phase_e_classic_multitarget.py`](../../scripts/run_phase_e_classic_multitarget.py)
- entrena dos modelos por algoritmo y fold: uno para `sleep_stage` y otro para `apnea_binary`
- cross-dataset principal: `MIT-BIH -> St. Vincent` y `St. Vincent -> MIT-BIH`

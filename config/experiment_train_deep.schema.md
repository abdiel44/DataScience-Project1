# Esquema de `experiment_deep*.yaml` (Fase E Deep)

Referencia para los presets deep de `Sleep-EDF 2013 + Fpz-Cz`.

| Clave | Tipo | Descripcion |
|---|---|---|
| `experiment_name` | string | Subcarpeta bajo `output.root` |
| `train_csv` | string | Tabla epoch-level con `source_file`, tiempos de epoca y target textual |
| `cross_dataset` | bool | Si `true`, usa `eval_csv` para evaluacion B |
| `eval_csv` | string \| null | Tabla del dominio destino |
| `subject_column` | string | Agrupacion subject-wise |
| `recording_column` | string | Columna por recording; no se cruzan sus limites al construir secuencias |
| `target_column` | string | Objetivo, tipicamente `sleep_stage` |
| `task` | `multiclass` | En esta v1 deep solo staging multiclase |
| `dataset.raw_root` | string | Carpeta `data/raw` o equivalente |
| `dataset.input_mode` | `raw` \| `epoch_store` | Fuente del waveform durante training |
| `dataset.epoch_store_root` | string \| null | Carpeta con `.npy` por grabacion cuando `input_mode=epoch_store` |
| `dataset.epoch_store_manifest` | string \| null | CSV/Parquet con `epoch_store_relpath` y `epoch_store_row` |
| `dataset.dataset_dirname` | string | Directorio del corpus dentro de `raw_root` |
| `dataset.signal_channel` | string | Canal EEG original a extraer |
| `dataset.sequence_length` | int impar | Longitud de secuencia; el target es la epoca central |
| `dataset.sample_hz` | int | Frecuencia objetivo de la senal |
| `dataset.epoch_seconds` | int | Duracion de cada epoca |
| `dataset.label_order` | lista | Orden preferido de clases para metricas |
| `model.type` | `cnn` \| `conformer` | Arquitectura supervisada |
| `model.*` | mapa | Dimensiones del encoder y del bloque temporal |
| `ssl.enabled` | bool | Activa pretraining contrastivo |
| `ssl.*` | mapa | Epocas, `projection_dim`, `temperature`, LR |
| `train.*` | mapa | `epochs`, `batch_size`, `lr`, `weight_decay`, `mixed_precision`, `class_weight`, etc. |
| `train.batching_strategy` | `random` \| `recording_blocked` | Estrategia de batches; `recording_blocked` recomendado con `epoch_store` |
| `augmentations.*` | mapa | Ruido, `amplitude scaling`, `time masking`, `frequency dropout` |
| `device.preferred` | string | `cuda` o `cpu` |
| `output.*` | mapa | Rutas y guardado de checkpoints |

Notas:
- Con `dataset.input_mode: raw`, el runner deep lee waveform desde los archivos originales referenciados por `source_file`; `train_csv` no necesita contener las 3000 muestras por epoca.
- Con `dataset.input_mode: epoch_store`, el runner deep lee epocas precomputadas desde `dataset.epoch_store_root` y `dataset.epoch_store_manifest`; si faltan, ejecutar [`scripts/materialize_epoch_store.py`](../scripts/materialize_epoch_store.py).
- Para `Sleep-EDF 2013 comparable`, el `train_csv` recomendado es `data/processed/sleep_edf_2013_fpzcz_raw.csv`.
- La salida mantiene la convencion de Fase E: `metrics_per_fold.csv`, `summary.json`, `predictions/`, `figures/`, `models/` y `config_resolved.yaml`.
- El script de entrada es [`scripts/run_phase_e_deep.py`](../scripts/run_phase_e_deep.py).

Multitask apnea:
- Para la linea apnea/no-apnea + staging auxiliar, usar [`scripts/prepare_multitask_apnea_metadata.py`](../scripts/prepare_multitask_apnea_metadata.py) y luego [`scripts/run_phase_e_multitask.py`](../scripts/run_phase_e_multitask.py).
- El CSV multitask debe incluir `dataset_id`, `subject_unit_id`, `recording_id`, `epoch_start_sec`, `epoch_end_sec`, `apnea_binary` y `sleep_stage` opcional.
- Para el pipeline optimizado, materializar antes el store con `scripts/materialize_epoch_store.py --config <yaml> --mode waveforms` y usar el manifiesto resultante en `dataset.epoch_store_manifest`.

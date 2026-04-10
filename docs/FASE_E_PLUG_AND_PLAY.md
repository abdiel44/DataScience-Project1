# Fase E — Ajustes plug-and-play (antes de “solo Enter”)

Este documento alinea el [PRD §4 y §11](PRD.md) con el runner [`train_runner`](../src/modeling/train_runner.py) y el enunciado del curso (PDF *Proyecto 1*): no basta con ejecutar el YAML por defecto; debes alinear **rutas**, **columnas**, **tipo de tarea** y, si aplica, **lista explícita de features** (EEG monocanal). Los **hiperparámetros** del YAML son valores razonables pero **no optimizados** para tu corpus; el PDF pide búsqueda de hiperparámetros **solo en train** (p. ej. por fold o en un notebook/Colab), sin tunear en el conjunto B en experimentos cross-dataset.

## 1. Rutas `train_csv` y `eval_csv`

El runner **comprueba que el archivo exista** antes de leer; si la ruta es relativa, intenta el cwd y **el directorio donde está el YAML** (útil con `config/experiments/foo.yaml` y `train_csv: ../data/out.csv`).

- Cross-dataset (PRD E3): `cross_dataset: true` y `eval_csv`. No se tunear en B.
- Presets [`config/experiments/`](../config/experiments/README.md): apuntan a `data/processed/...`. Esos CSV **no van en git**: debes generarlos (export WFDB, ingesta, etc.); si falta el archivo, el error indica las rutas probadas.

## 2. `subject_column` y `target_column`

Deben existir en el CSV y coincidir con tu pipeline de ingesta/export:

| Corpus (PRD) | `subject_column` típico | Objetivo típico (`target_column`) |
|--------------|-------------------------|-------------------------------------|
| MIT-BIH / SHHS (WFDB) | `record_id` | `sleep_stage` (estadificación) o columnas de eventos |
| ISRUC-Sleep | `subject_unit_id` | `event_group` o `sleep_stage` (según CSV) |
| St. Vincent | `recording_id` | `stage_mode` |
| Sleep-EDF (propia) | `recording_id` (o la que definas) | `sleep_stage` |

El runner excluye siempre `subject_column` y `target_column` de las features.

## 3. `task`: `binary` vs `multiclass`

- **`multiclass`**: estadificación u otras tareas con varias clases.
- **`binary`**: por defecto (`binary_require_zero_one_labels: true`) el runner **exige** valores **0 o 1** en la columna objetivo (número entero/float, bool, o strings `"0"`/`"1"`). Así las métricas tipo apnea (sensibilidad, especificidad, AUC) quedan alineadas con la matriz confusión 0 vs 1.
- Si necesitas dos etiquetas no numéricas, pon `binary_require_zero_one_labels: false` (sigue habiendo solo dos clases en train; en cross-eval, como mucho dos en eval).

Más de dos valores con `task: binary` → error al entrenar.

## 4. `feature_exclude` e `feature_include`

- **`feature_exclude`**: como antes.
- **`feature_include`**: si omites o dejas `null`, el runner usa todas las numéricas **salvo** que detecte **más de un prefijo** tipo export WFDB (`señal_mean`, `señal_std` con distintos stems). En ese caso **falla** y pide `feature_include` con **un solo** par stem (`*_mean`/`*_std`). Excepción explícita: `allow_multi_channel_features: true`.
- Cross-dataset: intersección numérica o lista compartida en ambos CSV.

## 5. Hiperparámetros en YAML

El bloque `hyperparams` es **fijo** en cada ejecución del runner. Para cumplir el PDF (búsqueda solo en train):

- Ajusta a mano tras pilotos, o
- Implementa grid/random search en un notebook **usando solo train** (idealmente dentro de cada fold de CV por sujeto para no sesgar la estimación).

El runner **no** sustituye ese trabajo de tuning.

## 6. Comando rápido

Desde la raíz del repo (o con `PYTHONPATH=src`):

```bash
python scripts/run_phase_e_cv.py --config config/experiments/mitbih_sleep_stages.yaml
```

Copia un preset a `config/mi_exp.yaml` si necesitas rutas absolutas (Colab) o cambios locales sin tocar el preset versionado.

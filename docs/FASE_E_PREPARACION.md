# Fase E (modelado): runner, Colab y utilidades

El [PRD §11](PRD.md) define la **Fase E** (E1 baselines, E2 tres familias de modelos, E3 subject-wise + cross-dataset, E4 métricas). Este documento describe lo implementado en el repo y cómo ejecutarlo en local o en **Google Colab** / máquina con más recursos.

## Dependencias

- `requirements.txt`: `scikit-learn`, `imbalanced-learn`, `scipy`, `xgboost`, `pyyaml`.
- Opcional: `hmmlearn` (HMM, E2).

## Orquestador principal: `train_runner`

[`src/modeling/train_runner.py`](../src/modeling/train_runner.py) lee un YAML (ver [`config/experiment_train.example.yaml`](../config/experiment_train.example.yaml), [`config/experiment_train.schema.md`](experiment_train.schema.md) y presets en [`config/experiments/`](../config/experiments/README.md)). Guía de ajustes obligatorios (rutas, columnas, binario apnea, tuning): [FASE_E_PLUG_AND_PLAY.md](FASE_E_PLUG_AND_PLAY.md). Comportamiento:

- Aplica `ensure_subject_unit_column` al CSV de entrenamiento.
- **Modo CV** (`cross_dataset: false`): validación cruzada **por sujeto** con SVM-RBF, Random Forest y XGBoost (activables en YAML); guarda predicciones, matrices de confusión PNG, `metrics_per_fold.csv` y `summary.json` bajo `output.root / experiment_name /`.
- **Modo cross-dataset** (`cross_dataset: true` + `eval_csv`): entrena con **todo** `train_csv` y evalúa en `eval_csv` **sin** ajustar hiperparámetros en B (solo los fijos del YAML).

Solo se usan columnas **numéricas** como features; excluye `target_column`, `subject_column` y `feature_exclude`. Con `feature_include` no vacío se fuerza la lista exacta de columnas (útil para un solo canal EEG).

### Comandos (cwd = raíz del repo, `PYTHONPATH=src` o usar script)

```bash
python scripts/run_phase_e_cv.py --config config/mi_experimento.yaml
```

Equivalente:

```bash
python -m modeling.train_runner --config config/mi_experimento.yaml
```

### Flujo sugerido en Google Colab

1. Subir un zip del repositorio o clonarlo; subir o montar Drive con tus CSV en `data/processed/`.
2. `pip install -r requirements.txt`
3. Copiar `config/experiment_train.example.yaml` → `config/mi_exp.yaml` y poner rutas absolutas a CSV si hace falta (`/content/drive/...`).
4. Ejecutar `!python scripts/run_phase_e_cv.py --config config/mi_exp.yaml` desde la raíz del proyecto.
5. Descargar la carpeta `reports/experiments/<experiment_name>/` (predicciones, figuras, métricas).

Los hiperparámetros del YAML son **fijos**; búsquedas grandes (grid/random) conviene añadirlas en un notebook aparte **solo sobre train** dentro de cada fold. El runner valida por defecto: CSV existente (cwd o carpeta del YAML), **binario = 0/1** si `binary_require_zero_one_labels: true`, y **un solo stem** `*_mean`/`*_std` salvo `feature_include` o `allow_multi_channel_features` (ver [FASE_E_PLUG_AND_PLAY.md](FASE_E_PLUG_AND_PLAY.md)).

## Módulo `src/modeling/` (bloques reutilizables)

| Módulo | Rol |
|--------|-----|
| [`train_runner.py`](../src/modeling/train_runner.py) | CV y cross-dataset desde YAML |
| [`cv_split.py`](../src/modeling/cv_split.py) | `subject_wise_fold_indices`, `SubjectFoldConfig` |
| [`metrics.py`](../src/modeling/metrics.py) | `apnea_binary_metrics`, `multiclass_sleep_metrics`, `mcnemar_exact`, `fold_metrics_summary` |
| [`artifacts.py`](../src/modeling/artifacts.py) | `save_predictions_dataframe`, `save_confusion_matrix_figure` |
| [`subject_id.py`](../src/modeling/subject_id.py) | `ensure_subject_unit_column` |

## Identificador de sujeto / registro

- WFDB (MIT-BIH, SHHS): **`record_id`**; St. Vincent / Sleep-EDF: **`recording_id`**.
- ISRUC: **`subject_unit_id`** derivado de la ruta de carpetas bajo `ISRUC-Sleep/` (sin nombre de archivo).
- Unificar con `ensure_subject_unit_column` si hace falta.

## E1 — Baselines bibliográficos

Plantilla para el informe: [BASELINES_LITERATURA_PLANTILLA.md](BASELINES_LITERATURA_PLANTILLA.md).

## Tests locales

- Por defecto: `pytest -q` (excluye pruebas marcadas `slow`).
- Incluir modelo pesado de smoke: `pytest -q -m slow`.

## Referencias cruzadas

- Comandos: [reproducir.txt](../reproducir.txt).
- PRD: [PRD.md](PRD.md) §11 Fase E (v1.7+).

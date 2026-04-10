# Esquema de `experiment_train*.yaml` (Fase E)

Referencia rápida para [`experiment_train.example.yaml`](experiment_train.example.yaml).

| Clave | Tipo | Descripción |
|-------|------|-------------|
| `experiment_name` | string | Subcarpeta bajo `output.root` |
| `random_seed` | int | Semillas sklearn / xgboost |
| `train_csv` | string | CSV de entrenamiento (obligatorio) |
| `cross_dataset` | bool | Si `true` y hay `eval_csv`, no se hace CV: fit en todo `train_csv`, métricas en `eval_csv` |
| `eval_csv` | string \| null | CSV del conjunto B (cross-dataset) |
| `subject_column` | string | Columna de agrupación (tras `ensure_subject_unit_column` si aplica) |
| `target_column` | string | Objetivo |
| `task` | `binary` \| `multiclass` | Familia de métricas |
| `binary_require_zero_one_labels` | bool | Si `true` (defecto) y `task: binary`, el target debe ser solo `0`/`1` (ver validación en `train_runner`). Si `false`, bastan dos clases cualesquiera |
| `allow_multi_channel_features` | bool | Si `false` (defecto), más de un prefijo entre columnas `*_mean`/`*_std` sin `feature_include` → error (EEG monocanal) |
| `feature_exclude` | lista | Columnas no usadas como features |
| `feature_include` | lista \| null | Si es lista no vacía, solo esas columnas numéricas; si `null` o `[]`, todas las numéricas tras excluir (cross-dataset: misma lógica; intersección train/eval si no hay include) |
| `cv.n_splits` | int | Pliegues subject-wise |
| `cv.stratify` | bool | `StratifiedGroupKFold` vs `GroupKFold` |
| `output.root` | string | Raíz de artefactos |
| `models.*` | bool | Activar SVM-RBF, RF, XGBoost |
| `hyperparams.*` | mapa | Parámetros fijos de cada modelo |

**Notas:** `train_csv` / `eval_csv` se buscan primero tal cual, luego bajo el cwd, luego bajo el directorio del archivo YAML. Solo se usan columnas **numéricas** como features. El CV usa índices train/test por sujeto; en cross-dataset no hay fugas de tuning en B porque los hiperparámetros vienen solo del YAML. Presets por corpus: [`config/experiments/README.md`](../config/experiments/README.md).

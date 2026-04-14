# Esquema de `experiment_train*.yaml` (Fase E)

Referencia rapida para [`experiment_train.example.yaml`](experiment_train.example.yaml).

| Clave | Tipo | Descripcion |
|-------|------|-------------|
| `experiment_name` | string | Subcarpeta bajo `output.root` |
| `random_seed` | int | Semillas sklearn / xgboost |
| `train_csv` | string | Tabla de entrenamiento `.csv` o `.parquet` (obligatoria) |
| `cross_dataset` | bool | Si `true` y hay `eval_csv`, no se hace CV: fit en todo `train_csv`, metricas en `eval_csv` |
| `eval_csv` | string \| null | Tabla del conjunto B `.csv` o `.parquet` (cross-dataset) |
| `subject_column` | string | Columna de agrupacion (tras `ensure_subject_unit_column` si aplica) |
| `target_column` | string | Objetivo |
| `task` | `binary` \| `multiclass` | Familia de metricas |
| `binary_require_zero_one_labels` | bool | Si `true` (defecto) y `task: binary`, el target debe ser solo `0`/`1` |
| `allow_multi_channel_features` | bool | Si `false` (defecto), mas de un prefijo entre columnas `*_mean`/`*_std` sin `feature_include` -> error |
| `feature_exclude` | lista | Columnas no usadas como features |
| `feature_include` | lista \| null | Si es lista no vacia, solo esas columnas numericas; si `null` o `[]`, todas las numericas tras excluir |
| `cv.n_splits` | int | Pliegues subject-wise |
| `cv.stratify` | bool | `StratifiedGroupKFold` vs `GroupKFold` |
| `train_resampling.*` | mapa | Resampling train-only opcional; por ahora `smote_to_reference_minus` |
| `output.root` | string | Raiz de artefactos |
| `output.resume_completed` | bool | Si `true`, salta folds ya completos si existen sus artefactos |
| `models.*` | bool | Activar SVM-RBF, RF, XGBoost |
| `hyperparams.*` | mapa | Parametros fijos de cada modelo |
| `tuning.*` | mapa | Nested CV opcional, search space y submuestreo de sujetos para tuning |

**Notas:** `train_csv` / `eval_csv` se buscan primero tal cual, luego bajo el cwd, luego bajo el directorio del archivo YAML. La lectura usa la extension del archivo para elegir `read_csv` o `read_parquet`. Solo se usan columnas numericas como features y se convierten a `float32` antes del fit. Para `svm_rbf`, el escalado se ajusta fuera del estimador para que cualquier resampling ocurra solo sobre `outer-train`. El CV usa indices train/test por sujeto; en cross-dataset no hay fugas de tuning en B porque los hiperparametros se seleccionan solo en `train_csv`. Presets por corpus: [`config/experiments/README.md`](../config/experiments/README.md).

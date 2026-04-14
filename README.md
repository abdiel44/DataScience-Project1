# Data Science Project Template (Python + Docker)

Plantilla base para cursos/proyectos de Data Science donde el dataset final lo entrega el profesor.

## Objetivo

Este template permite:
- Cargar un dataset (`.csv`) desde `data/raw/`
- Ejecutar un pipeline de preprocesamiento
- Guardar datos limpios en `data/processed/`
- Ejecutar todo localmente o dentro de Docker

## Estructura del proyecto

```text
.
├── data/
│   ├── raw/            # datasets originales (no versionar archivos grandes)
│   └── processed/      # datasets procesados
├── notebooks/          # notebooks de exploración
├── reports/            # salidas, métricas, figuras
├── src/
│   ├── main.py         # orquesta pasos opcionales de preprocesamiento y EDA
│   ├── cleaning.py     # limpieza de datos
│   ├── encoding.py     # Topic 1: nominal / ordinal / binarios
│   ├── class_balance.py  # Topic 11: sub/sobremuestreo, SMOTE, pesos sugeridos
│   ├── scaling.py      # Topic 5: normalización min-max o estandarización z-score
│   ├── dimensionality.py  # Topic 6: PCA, LDA, varianza, SelectKBest
│   └── eda.py          # análisis exploratorio (sobre datos limpios)
├── tests/              # pruebas automáticas
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Requisitos

- Docker Desktop (recomendado para todos los estudiantes)
- Opcional: Python 3.11+ para ejecutar sin Docker

## Uso con Docker (recomendado)

1. Copia tu dataset a `data/raw/`, por ejemplo: `data/raw/profesor_dataset.csv`
2. Construye la imagen:

```bash
docker compose build
```

3. Ejecuta el pipeline:

```bash
docker compose run --rm app python src/main.py --input data/raw/profesor_dataset.csv --output data/processed/dataset_limpio.csv
```

## Uso local (sin Docker)

1. Crear entorno virtual e instalar dependencias:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Ejecutar:

```bash
python src/main.py --input data/raw/profesor_dataset.csv --output data/processed/dataset_limpio.csv
```

## EDA para EEG (sueño + apnea)

Además de limpiar el dataset, el pipeline puede ejecutar una etapa de EDA que genera tablas, figuras y un resumen en Markdown para tu informe.

- **Fase B (datos brutos / tabular exportado):** [`scripts/raw_eda.py`](scripts/raw_eda.py) lee un CSV, normaliza nombres de columnas a snake_case y escribe en `reports/eda_raw/<task>/` **sin** pasar por limpieza, codificación ni escalado de `main.py`. Úsalo para caracterizar exports WFDB→CSV u otras tablas antes del preprocesamiento completo (ver [docs/EDA_METODOLOGIA.md](docs/EDA_METODOLOGIA.md)).
- **Exportar tabla raw desde loaders `--source`:** [`scripts/export_source_raw.py`](scripts/export_source_raw.py) materializa un CSV tabular inicial para ISRUC, St. Vincent, Sleep-EDF u otras fuentes soportadas antes de pasar por `raw_eda.py` o por `main.py`.
- **Tras el pipeline de `main.py`:** con `--run-eda`, el EDA corre por defecto sobre el dataframe **solo limpio** (sin codificar ni escalar); la salida va a `--eda-outdir` o `reports/eda/<task>/`.
- **Fase D (datos ya preprocesados):** usa `--run-eda` junto con **`--run-eda-processed`** para analizar la tabla **final** (tras encoding, balanceo, escalado, dimensionalidad). Salida por defecto: `reports/eda_processed/<task>/`. Detalle: [docs/EDA_POST_PREPROCESAMIENTO.md](docs/EDA_POST_PREPROCESAMIENTO.md).

### Ejemplo: clasificación de etapas de sueño

```bash
python src/main.py --input data/raw/sleep_dataset.csv --output data/processed/sleep_clean.csv --run-eda --task sleep --target-col sleep_stage --eda-outdir reports/eda/sleep_staging
```

### Ejemplo: detección de apnea

```bash
python src/main.py --input data/raw/apnea_dataset.csv --output data/processed/apnea_clean.csv --run-eda --task apnea --target-col apnea_label --eda-outdir reports/eda/apnea_detection
```

### Archivos de salida EDA (por tarea)

En el directorio de salida indicado (`--eda-outdir`) se generan:

- `01_dataset_profile.csv`: perfil de columnas, tipos, missing y cardinalidad
- `02_descriptive_numeric.csv`: media, mediana, moda, dispersión e indicadores de forma
- `03_descriptive_categorical.csv`: frecuencias y proporciones por categoría
- `04_correlations_pearson.csv`: matriz de correlaciones de Pearson
- `05_correlations_spearman.csv`: matriz de correlaciones de Spearman
- `fig_hist_<feature>.png` y `fig_box_<feature>.png`: histogramas y boxplots de variables numéricas
- `fig_corr_heatmap.png`: heatmap de correlaciones
- `fig_target_distribution.png`: distribución de clases de la variable objetivo
- `eda_summary.md`: resumen interpretativo breve para el informe final

## Qué hace `main.py`

**Limpieza (`cleaning.py`):** filas vacías / target faltante, deduplicar, texto, coercionar numéricos, winsorización Tukey, imputación, etc. Con `--write-cleaning-report`, artefactos en `reports/cleaning/...`.

**Codificación opcional (`encoding.py`):** con `--encoding-spec mi_spec.json` declaras columnas **nominales** (one-hot), **ordinales** (orden explícito → enteros) y **binarias** (0/1). `--write-encoding-report` genera `encoding_summary.md`.

**Equilibrio de clases opcional (`class_balance.py`, Topic 11):** después de codificar (recomendado para **SMOTE**, que exige features numéricas) y **antes** del escalado. Usa `--balance-method random_under|random_over|smote` (requiere `--target-col`), o define `class_balance_method` en el JSON de `--encoding-spec` si `--balance-method none`. Opciones: `--balance-strategy` (p. ej. JSON para `sampling_strategy` de imbalanced-learn), `--smote-k-neighbors` si la minoría es pequeña, `--write-class-balance-report`. El informe incluye pesos `class_weight` sugeridos (distribución pre-remuestreo) y recordatorios de métricas (F1, AUC-PR, stratify).

**Escalado opcional (`scaling.py`, Topic 5):** después del balanceo de clases. Usa `--scale-method standardize` o `minmax`, o `numeric_scaling` en el JSON si `--scale-method none`. `--write-scaling-report` en `reports/scaling/...`.

**Reducción / selección de features (`dimensionality.py`, Topic 6):** opcional, **después del escalado** (PCA/LDA suelen ir sobre features ya escaladas si el flujo incluye escalado). Métodos: `pca`, `lda`, `variance_threshold`, `select_k_best`; CLI (`--dimensionality-method`, `--pca-n-components`, `--lda-n-components`, `--variance-threshold`, `--select-k`, `--select-score-func`, `--dimensionality-exclude`, `--write-dimensionality-report`) o claves en el JSON de `--encoding-spec` si el método CLI es `none`. Para **LDA** y **SelectKBest** hace falta `--target-col` (o `dimensionality_target_column` en JSON).

| Enfoque | Rol típico |
|--------|------------|
| **PCA** | Sin etiquetas: comprime varianza, componentes menos interpretables que las features originales. |
| **LDA** | Con clases: maximiza separación entre clases; como máximo `n_clases - 1` componentes. |

El CSV de `--output` refleja **limpieza → codificación (si aplica) → balance de clases (si aplica) → escalado (si aplica) → dimensionalidad (si aplica)**.

**EDA (`eda.py`), opcional con `--run-eda`:** por defecto sobre el dataframe **solo limpio** (sin codificar ni escalar). Con **`--run-eda-processed`** (Fase D), el EDA usa la tabla **final** del pipeline; salida por defecto `reports/eda_processed/<task>/`.

Ejemplo de `mi_spec.json` (las claves de escalado las lee `scaling.py`; `encoding.py` las ignora al codificar):

```json
{
  "nominal_columns": ["city", "channel"],
  "ordinal_columns": { "education_level": ["primary", "secondary", "tertiary"] },
  "binary_columns": ["is_male"],
  "drop_first_dummy": false,
  "numeric_scaling": "standardize",
  "scale_exclude": ["subject_id"],
  "scale_include": null,
  "target_column": "sleep_stage",
  "class_balance_method": "none",
  "smote_k_neighbors": null,
  "dimensionality_method": "none",
  "pca_n_components": 0.95,
  "dimensionality_target_column": "sleep_stage",
  "dimensionality_feature_exclude": []
}
```

Para activar balanceo solo vía JSON (sin columnas de codificación), usa un `--encoding-spec` mínimo con `class_balance_method` y `target_column` / `--target-col`.

## Pruebas

```bash
pytest -q
```

Por defecto se excluyen pruebas marcadas `slow`. Para incluir el smoke de Fase E con los tres modelos: `pytest -q -m slow`.

## Alcance experimental y reproducibilidad (Fase A)

- **Alcance y etiquetas:** [docs/ALCANCE_EXPERIMENTAL.md](docs/ALCANCE_EXPERIMENTAL.md) y [config/experiment_scope.yaml](config/experiment_scope.yaml).
- **Comandos de ejemplo:** [reproducir.txt](reproducir.txt) (requisito del curso).
- **Semillas:** [config/reproducibility.env.example](config/reproducibility.env.example).
- **Uso de IA (plantilla para el informe):** [docs/USO_IA_PLANTILLA.md](docs/USO_IA_PLANTILLA.md).

## Preprocesamiento y configuraciones (Fase C)

- **Diseño metodológico (ventanas, escalado, límites del pipeline):** [docs/DISENO_PREPROCESAMIENTO.md](docs/DISENO_PREPROCESAMIENTO.md).
- **Índice de JSON en `config/datasets/` y comando con `--encoding-spec`:** [docs/CONFIG_DATASETS.md](docs/CONFIG_DATASETS.md).

## EDA tras preprocesamiento (Fase D)

- **Metodología y comparación antes/después:** [docs/EDA_POST_PREPROCESAMIENTO.md](docs/EDA_POST_PREPROCESAMIENTO.md).
- **Plantilla de texto para el informe:** [docs/DATOS_TRAS_PREPROCESAMIENTO_PLANTILLA.md](docs/DATOS_TRAS_PREPROCESAMIENTO_PLANTILLA.md).

## Modelado (Fase E): CV y cross-dataset

- **Guía (incl. Google Colab):** [docs/FASE_E_PREPARACION.md](docs/FASE_E_PREPARACION.md).
- **Ejecutar experimento:** copiar [config/experiment_train.example.yaml](config/experiment_train.example.yaml), ajustar rutas y columnas, luego:
  - `python scripts/run_phase_e_cv.py --config config/mi_experimento.yaml`
  - o `python -m modeling.train_runner --config ...` con `PYTHONPATH=src`.
- **Salidas:** `reports/experiments/<experiment_name>/` (predicciones, matrices de confusión, `metrics_per_fold.csv`, `summary.json`).
- **Baselines (E1):** plantilla [docs/BASELINES_LITERATURA_PLANTILLA.md](docs/BASELINES_LITERATURA_PLANTILLA.md).
- **Columna de sujeto:** ver guía; usar `ensure_subject_unit_column` si hace falta.

## Sugerencias para estudiantes

- Mantener notebooks para exploración y mover lógica final a `src/`
- Separar etapas: limpieza, features, entrenamiento, evaluación
- Versionar código, no datasets pesados
- Documentar decisiones y supuestos

## Runner multitarget clÃ¡sico

- `python scripts/run_phase_e_classic_multitarget.py --config config/experiments/mitbih_apnea_stage_classic.yaml`
- `python scripts/run_phase_e_classic_multitarget.py --config config/experiments/cross_dataset_mitbih_to_st_vincent_classic.yaml`
- `python scripts/run_phase_e_classic_multitarget.py --config config/experiments/cross_dataset_st_vincent_to_mitbih_classic.yaml`

## Informe final en LaTeX

- **RaÃ­z del informe:** [report/latex/main.tex](report/latex/main.tex)
- **GeneraciÃ³n de assets desde experimentos reales:** `python scripts/build_latex_report_assets.py`
- **CompilaciÃ³n manual del PDF:**
  - `cd report/latex`
  - `pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex`
  - `bibtex main`
  - `pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex`
  - `pdflatex --disable-installer -interaction=nonstopmode -halt-on-error main.tex`
- **Script de conveniencia (PowerShell):** `.\report\latex\build_report.ps1`
- **Salida esperada:** `report/latex/main.pdf`


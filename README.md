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

**EDA (`eda.py`), opcional con `--run-eda`:** sobre el dataframe **solo limpio** (sin codificar ni escalar), para tablas y gráficos interpretables.

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

## Sugerencias para estudiantes

- Mantener notebooks para exploración y mover lógica final a `src/`
- Separar etapas: limpieza, features, entrenamiento, evaluación
- Versionar código, no datasets pesados
- Documentar decisiones y supuestos


# EDA sobre datos brutos / tabulares (Fase B — B2 y B3)

## B2. Qué genera el repositorio

Para cualquier tabla CSV con columna objetivo, el módulo [`src/pre_processing/eda.py`](../src/pre_processing/eda.py) puede producir:

- **Opción A (recomendada para Fase B):** script [`scripts/raw_eda.py`](../scripts/raw_eda.py) — carga el CSV, normaliza nombres de columnas a snake_case y escribe en `reports/eda_raw/<task>/` sin ejecutar el resto del pipeline de [`src/main.py`](../src/main.py).
- **Opción B:** `python src/main.py ... --run-eda` sobre un CSV ya en la ruta deseada (pasa por limpieza/orquestación según flags).

Salidas típicas (A o B, según configuración):

- `01_dataset_profile.csv` — tipos, missing, cardinalidad por columna  
- `02_descriptive_numeric.csv` — estadísticos por variable numérica (media, std, IQR, sesgo, curtosis)  
- `03_descriptive_categorical.csv` — frecuencias por categoría  
- `04_correlations_pearson.csv` / `05_correlations_spearman.csv`  
- Figuras PNG: histogramas, boxplots, heatmap de correlación, distribución del **target**  
- `eda_summary.md` — resumen textual  

**Énfasis para el informe:** revisar frecuencias de clase (N1 suele ser minoritaria en staging; apnea/eventos minoritarios en detección).

## B3. Transparencia metodológica (obligatorio en el texto)

1. **EDA exploratorio global** (sobre una tabla conveniente para entender el corpus) **no sustituye** la evaluación final del modelo. Las métricas oficiales (accuracy, F1, kappa, AUC, etc.) deben calcularse solo sobre **particiones train/validación** definidas con protocolo **subject-wise**, sin mezclar sujetos entre train y test.

2. Si en un análisis preliminar se usó un split incorrecto (p. ej. filas aleatorias sin agrupar por sujeto), documentarlo como **limitación** o **diseño preliminar**, no como resultado final.

3. El párrafo “Caracterización de los datos brutos” del informe (1–2 páginas) puede apoyarse en las tablas/figuras de `reports/eda_raw/...`, citando rutas de artefactos.

---

## Comando rápido

- Fase B (EDA crudo / tabular): ver [reproducir.txt](../reproducir.txt) (línea Fase B) o el PRD §11.
- EDA tras preprocesamiento: `python src/main.py ... --run-eda --eda-outdir reports/eda/<task>`

# Plantilla narrativa — Datos tras el preprocesamiento (informe)

*Completar en el documento final. Eliminar instrucciones entre corchetes.*

## 1. Resumen del flujo aplicado

[Indicar qué pasos se activaron: limpieza, codificación, balanceo, escalado, PCA/LDA/SelectKBest, según `main.py` y el JSON de `config/datasets/`. Referenciar rutas de reportes en `reports/cleaning`, `reports/encoding`, etc.]

## 2. Comparación antes / después (D2)

**Distribución de la variable objetivo:** [Describir cambios respecto a la caracterización bruta — p. ej. tras `random_under`, SMOTE, o sin balanceo. Citar `fig_target_distribution.png` en `reports/eda_raw/` vs `reports/eda_processed/`.]

**Dimensionalidad:** [Número de columnas o features antes del one-hot/PCA y después.]

**Calidad tabular:** [Missingness, correlaciones fuertes tras escalado; advertir si el target quedó en variables dummy y cómo se interpreta el EDA.]

## 3. Relación con el modelado

[Indicar que esta tabla (`--output` de `main.py`) es la base para entrenamiento; la validación subject-wise se aplica en la etapa de modelos, no en este EDA global.]

---

*Coherente con [EDA_POST_PREPROCESAMIENTO.md](EDA_POST_PREPROCESAMIENTO.md) y el PRD Fase D.*

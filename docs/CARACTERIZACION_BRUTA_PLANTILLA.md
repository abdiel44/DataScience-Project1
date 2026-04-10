# Plantilla narrativa — Caracterización de los datos brutos (informe)

*Completar 1–2 páginas en el documento final. Eliminar instrucciones entre corchetes.*

## 1. Origen y composición

[Describir de dónde provienen los datos, número aproximado de registros/sujetos, y si se trabaja con épocas de 30 s o con otra unidad. Referenciar `docs/DATASET_INVENTORY.md`.]

## 2. Variable objetivo

**Estadificación:** [distribución de W, N1, N2, N3, REM; comentar el desafío de N1 si aparece minoritaria.]

**Apnea / eventos:** [definición usada; desbalance observado.]

## 3. Calidad y distribución de features

[Missingness global; variables numéricas con mayor dispersión o sesgo; correlaciones fuertes que sugieran redundancia.]

## 4. Limitaciones observadas antes del modelado

[Ruido, heterogeneidad entre cohortes, ausencia de identificador de sujeto en alguna tabla, etc.]

## 5. Relación con el protocolo de evaluación

Esta caracterización se basa en un vistazo **exploratorio** de los datos disponibles. La evaluación final de los modelos se realizará con validación **subject-wise** y, cuando aplique, experimentos **cross-dataset**, sin ajustar hiperparámetros en el conjunto de prueba destino.

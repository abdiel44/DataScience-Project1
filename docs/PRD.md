# PRD — Proyecto 1: Machine Learning para clasificación de etapas de sueño y detección de apnea

**Versión:** 1.7  
**Fuente de requisitos académicos:** [docs/Proyecto-1.pdf](Proyecto-1.pdf)  
**Repositorio:** plantilla de Ciencia de Datos (Python + Docker) con pipeline de preprocesamiento en `src/`

---

## 1. Resumen ejecutivo

Este documento define el **Producto** (entregables de software, datos y documentación) y los **Requisitos** alineados con el enunciado del curso: entrenar y comparar modelos tradicionales de ML (SVM-RBF, Random Forest / bagging, boosting) sobre **EEG monocanal** para (1) **estadificación del sueño** y (2) **detección de apnea**, con validación **subject-wise**, experimentos **cross-dataset**, comparación con **baselines** de literatura y propuesta de mejora metodológica.

El repositorio actual proporciona una **base reproducible** para la fase de **datos tabulares**: ingesta desde CSV o desde corpora WFDB (exportación a CSV por épocas de 30 s), limpieza, codificación, escalado, reducción de dimensionalidad, balanceo de clases y EDA.

---

## 2. Objetivos del producto (desde el PDF)

### 2.1 Objetivo general

Aplicar tres familias de algoritmos — **SVM con kernel (optimizar C, gamma, multiclase, class_weight)**, **Random Forest** y **boosting** (AdaBoost / XGBoost / Gradient Boosting) — a:

1. Clasificación de **etapas de sueño** (AASM: W, N1, N2, N3, REM u equivalentes documentados).
2. **Detección de apnea** (y métricas de severidad según el dataset), usando **solo EEG de un canal** (frontal o central estándar), comparando con trabajos publicados sobre los **mismos conjuntos de datos**.

### 2.2 Objetivos específicos (trazables en el PRD)

| ID | Objetivo | Evidencia esperada |
|----|----------|-------------------|
| O1 | Pipelines reproducibles: preprocesamiento → features → entrenamiento → evaluación | Código versionado, Docker, `reproducir.txt`, seeds registradas |
| O2 | Al menos tres familias de modelos implementadas y comparadas | Tablas CSV + figuras PNG por experimento |
| O3 | Validación **subject-wise k-fold** dentro de cada dataset | Particiones por sujeto sin fuga |
| O4 | Generalización **cross-dataset** (entrenar en A, probar en B sin tunear en test) | Experimentos documentados |
| O5 | 6–10 referencias científicas + ≥3 baselines comparables | Sección de informe + tabla comparativa |
| O6 | Análisis crítico + **propuesta de mejora** (teoría, pseudocódigo; implementación = bono) | Sección dedicada en informe |

### 2.3 Modelos y extensiones opcionales (PDF)

- **SVM (RBF):** hiperparámetros C, gamma, `class_weight`, estrategia multiclase; favorece features **normalizados/seleccionados**.
- **Random Forest / bagging:** `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`; reportar **importancia de features**.
- **Boosting:** comparar con RF bajo el **mismo protocolo**; útil ante **desbalance**.
- **HMM (opcional):** sobre features y/o como suavizado de secuencias; interpretar matriz de transición.

---

## 3. Alcance funcional del software (estado actual del proyecto)

### 3.1 Orquestación — [`src/main.py`](../src/main.py)

Flujo soportado (orden lógico):

1. **Entrada de datos:** CSV (`--input`) o ingesta desde `data/raw` (`--source`).
2. **Limpieza** — [`pre_processing/cleaning.py`](../src/pre_processing/cleaning.py): nombres en snake_case, duplicados, imputación, coercion numérica, outliers opcionales (Tukey), etc.
3. **Codificación (opcional)** — [`encoding.py`](../src/pre_processing/encoding.py): nominal (one-hot), ordinal, binarios; spec JSON.
4. **Balanceo de clases (opcional)** — [`class_balance.py`](../src/pre_processing/class_balance.py): under/over/SMOTE; pesos sugeridos.
5. **Escalado (opcional)** — [`scaling.py`](../src/pre_processing/scaling.py): estandarización o min-max (relevante para SVM).
6. **Dimensionalidad (opcional)** — [`dimensionality.py`](../src/pre_processing/dimensionality.py): PCA, LDA, umbral de varianza, SelectKBest.
7. **Salida:** CSV en `data/processed/`.
8. **EDA (opcional)** — [`eda.py`](../src/pre_processing/eda.py): perfiles, correlaciones, figuras PNG, resumen Markdown. Por defecto con `--run-eda` se analiza la tabla **solo limpia** (antes de codificar); con `--run-eda --run-eda-processed` (Fase D) se analiza la salida **final** del pipeline; ver [docs/EDA_POST_PREPROCESAMIENTO.md](EDA_POST_PREPROCESAMIENTO.md).

**Exportación WFDB → CSV (30 s):** `--export-epochs {mit-bih-psg,shhs-psg}` con `--output-stages` y `--output-events` — [`wfdb_epoch_export.py`](../src/pre_processing/wfdb_epoch_export.py).

### 3.2 Mapeo temático (curso ↔ módulos)

| Tema (DataScienceTopics / curso) | Módulo |
|----------------------------------|--------|
| Topic 1 — codificación de variables | `encoding.py` |
| Topic 5 — normalización / estandarización | `scaling.py` |
| Topic 6 — selección / extracción de features | `dimensionality.py` |
| Topic 11 — clases desbalanceadas | `class_balance.py` |
| EDA / perfilado | `eda.py` |
| Limpieza estructurada | `cleaning.py` |

### 3.3 Fuera de alcance del repositorio actual (gap respecto al PDF)

- **Modelado ML** (SVM, RF, XGBoost, HMM): no implementado en esta plantilla; es responsabilidad del equipo según el PRD académico.
- **Extracción de features de EEG crudo** (espectro, wavelets, etc.): el pipeline actual opera sobre **tablas** (features ya agregadas por época en CSV o derivadas del export WFDB: media/desviación por canal y ventana).
- **Validación subject-wise y cross-dataset:** deben implementarse en notebooks o scripts de entrenamiento con **particiones explícitas por sujeto**.

---

## 4. Datasets y uso previsto

### 4.1 Tabla alineada con [Proyecto-1.pdf](Proyecto-1.pdf)

| Dataset | Rol sugerido (PDF) | Etiquetas / notas | Soporte en este repo |
|---------|-------------------|-------------------|------------------------|
| **Sleep-EDF Expanded** | Base principal desarrollo / baseline | W/N1/N2/N3/REM; apnea si disponible | Ingesta resumen opcional (`--source sleep-edf-expanded`, requiere `mne`); o CSV propio |
| **MIT-BIH PSG** | Complementario apnea/sueño | Apnea/no-apnea por ventana; estadios en `.st` | `--export-epochs mit-bih-psg` → dos CSV (estadios + eventos respiratorios); `--source mit-bih-psg` |
| **SHHS** | Generalización / dominio | Apnea, severidad, staging según disponibilidad | `--export-epochs shhs-psg`; `--source shhs-psg` |
| **St. Vincent’s** | Clínica adicional | Apnea/staging según subconjunto | `--source st-vincent-apnea` (resúmenes desde `*_stage.txt`) |
| **ISRUC-Sleep** | Complementario staging | W/N1/N2/N3/REM | `--source isruc-sleep` (segmentos CSV → fila por archivo + features) |

### 4.2 Requisito académico: EEG monocanal

El PDF exige **un solo canal EEG** (frontal o central estándar). Los exports WFDB actuales incluyen **varias señales** por registro (ECG, respiración, etc.) para compatibilidad con PSG completo; el equipo debe **filtrar columnas** al canal EEG acordado y documentarlo en el informe (coherente con la rúbrica).

### 4.3 Configuración JSON de ejemplo

En [`config/datasets/`](../config/datasets/) existen especificaciones de encoding/scaling para: ISRUC, St. Vincent, MIT-BIH (staging/eventos), SHHS (staging/eventos), Sleep-EDF Expanded (plantilla). Sirven como plantilla para `--encoding-spec`; el índice actualizado está en [docs/CONFIG_DATASETS.md](CONFIG_DATASETS.md).

---

## 5. Requisitos no funcionales (PDF + plantilla)

### 5.1 Reproducibilidad

- **Docker obligatorio** para entornos reproducibles ([`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml)).
- Resultados tabulares en **CSV**; figuras finales en **PNG**.
- Registrar **seeds**, hiperparámetros, tiempos y particiones.
- **`reproducir.txt`** (o equivalente) con comandos de ejemplo.
- **`README.md`**: dependencias, estructura, datasets, pasos de reproducción.

### 5.2 Uso de IA (PDF)

- Permitido para exploración, resúmenes, código y depuración.
- Prohibido inventar referencias, métricas o experimentos no ejecutados.
- Declarar herramientas de IA usadas y conservar prompts/notas (apéndice o bitácora).

### 5.3 Entregables (PDF)

1. Informe final PDF.  
2. Scripts de ejemplos de ejecución.  
3. Código en `.zip` reproducible con Docker.  
4. README completo.

---

## 6. Métricas y análisis (criterios de aceptación del informe)

### 6.1 Apnea / no-apnea

- Accuracy, sensibilidad, especificidad, **AUC-ROC**; exactitud de severidad por paciente donde aplique.

### 6.2 Estadificación

- Accuracy global, **macro-F1**, **kappa de Cohen**; análisis por clase con énfasis en **N1**.

### 6.3 Estadísticos

- Media y desviación estándar sobre folds; IC bootstrap; **McNemar** y comparación de AUC cuando proceda.

*(El pipeline de preprocesamiento no calcula estas métricas; deben implementarse en la fase de modelado.)*

---

## 7. Estructura del informe (checklist PDF)

1. Portada  
2. Resumen 150–250 palabras  
3. Introducción  
4. Revisión de literatura (6–10 refs): preprocesamiento + baselines comparables  
5. Datasets y preprocesamiento (+ extracción de features)  
6. Modelos baseline: arquitectura, hiperparámetros, CV, resultados  
7. Resultados: tablas/figuras vs baselines externos  
8. Discusión (preguntas guía del PDF)  
9. Propuesta de mejora (teoría, diagrama, pseudocódigo) — implementación bono  
10. Conclusiones y trabajo futuro  
11. Referencias IEEE  
12. Apéndice: comandos (opcional)

---

## 8. Rúbrica (resumen ponderado — PDF)

| Criterio | Puntos |
|----------|--------|
| Baselines comparables | 20 |
| Diseño experimental (subject-wise, cross-dataset) | 10 |
| Resultados y métricas (CSV/PNG, estadística) | 10 |
| Análisis crítico (preguntas guía) | 30 |
| Propuesta de mejora | 10 |
| Reproducibilidad y documentación | 20 |
| **Bono:** implementación propuesta de mejora | hasta 20 |

---

## 9. Riesgos y dependencias

- **Cambio de dominio** entre cohortes (SHHS vs clínica vs público) — mitigar con protocolo cross-dataset explícito.  
- **N1** y clases minoritarias — usar métricas adecuadas y Topic 11 (balanceo/pesos).  
- **Datos crudos voluminosos** — no versionar en git; usar `data/raw/` ignorado salvo `.gitkeep`.  
- **Single-channel EEG** — alinear columnas exportadas con el canal elegido antes del modelado.

---

## 10. Glosario breve

- **Subject-wise:** ningún sujeto comparte train y test.  
- **Cross-dataset:** entrenar en un corpus y evaluar en otro sin ajustar hiperparámetros en el test destino.  
- **Baseline:** método publicado comparable en datos y protocolo.  
- **PRD:** Product Requirements Document — este documento.

---

## 11. Roadmap: pasos restantes hasta un informe tipo artículo científico

Esta sección define el **camino de trabajo** para completar el proyecto y redactar un informe profesional que: (1) caracterice los **datos brutos** (EDA), (2) documente el **preprocesamiento**, (3) muestre los datos **después del preprocesamiento**, y (4) continúe con modelado, resultados y discusión según *Proyecto-1.pdf* y la rúbrica.

**Narrativa recomendada del documento final:** datos brutos (EDA) → métodos (preprocesamiento y features) → datos procesados (EDA) → modelos y validación → resultados → discusión (preguntas guía) → propuesta de mejora → conclusiones.

### Fase A — Alcance, contratos de datos y reproducibilidad

| Paso | Detalle |
|------|---------|
| **A1. Fijar el alcance experimental** | **Tarea 1 (estadificación):** conjunto de etiquetas (p. ej. W/N1/N2/N3/REM) y **mapeo** desde cada corpus (documentar excepciones). **Tarea 2 (apnea):** binario vs severidad; qué corpus soporta qué objetivo; ventanas coherentes con anotaciones (p. ej. 30 s). **EEG monocanal:** elegir **un** canal por dataset (frontal o central); nombre exacto de columna/señal tras export/ingesta. |
| **A2. Línea base de reproducibilidad** | Versiones de Python y dependencias (p. ej. pin en `requirements.txt`). Archivo **`reproducir.txt`**: layout de datos, entorno, Docker, comandos de preprocesamiento y (cuando existan) entrenamiento; **seeds** y commit git registrados. |
| **A3. Ética / IA (curso)** | Subsección breve “Uso de IA”: herramientas, qué se verificó manualmente; prompts o bitácora en apéndice. No inventar referencias ni métricas. |

**Estado Fase A (v1.2):** completada en el repositorio con artefactos enlazados.

| Entregable | Ubicación |
|------------|-----------|
| A1 Alcance y contrato de datos (narrativa + política EEG monocanal) | [docs/ALCANCE_EXPERIMENTAL.md](ALCANCE_EXPERIMENTAL.md) |
| A1 Mapeo estructurado (YAML) | [config/experiment_scope.yaml](../config/experiment_scope.yaml) |
| A2 Comandos de reproducción | [reproducir.txt](../reproducir.txt) en la raíz |
| A2 Semillas de ejemplo | [config/reproducibility.env.example](../config/reproducibility.env.example) |
| A3 Plantilla uso de IA (informe) | [docs/USO_IA_PLANTILLA.md](USO_IA_PLANTILLA.md) |

*Pendiente solo de evolución del proyecto:* fijar definitivamente el canal EEG por dataset tras pilotos y actualizar `experiment_scope.yaml` + informe en bloque.

### Fase B — Comprensión de datos brutos (EDA “cómo se ven”)

| Paso | Detalle |
|------|---------|
| **B1. Inventario por dataset** | Tabla: nombre, fuente, #sujetos o registros, duración, fs, canales, tipo de anotación, enlace a documentación. Aplicar a cada corpus usado (Sleep-EDF, MIT-BIH, SHHS, St. Vincent, ISRUC según alcance). |
| **B2. EDA sobre crudo / tabular inicial** | Univariado: missingness, tipos, **frecuencias de clase** (énfasis N1 y apnea minoritaria). Resúmenes numéricos (media, std, IQR, sesgo). **Figuras PNG** para el informe: distribución de clases, correlaciones si aplica, fragmentos temporales si se grafica EEG. |
| **B3. Transparencia metodológica** | Declarar que un EDA exploratorio sobre “todo el conjunto” es distinto del **test final**; las métricas finales deben venir solo de **train/validación** por protocolo subject-wise. Párrafo narrativo “Caracterización de los datos brutos” (1–2 páginas). |

**Estado Fase B (v1.3):** documentación, plantillas y **CLI** `scripts/raw_eda.py` en el repositorio para B2 sin depender del pipeline completo de `main.py` (ver tabla siguiente).

| Entregable | Ubicación |
|------------|-----------|
| B1 Inventario por dataset (tabla editable con cifras locales) | [docs/DATASET_INVENTORY.md](DATASET_INVENTORY.md) |
| B2/B3 Metodología EDA crudo + disclaimer subject-wise | [docs/EDA_METODOLOGIA.md](EDA_METODOLOGIA.md) |
| B3 Narrativa “Caracterización de los datos brutos” (informe) | [docs/CARACTERIZACION_BRUTA_PLANTILLA.md](CARACTERIZACION_BRUTA_PLANTILLA.md) |
| B2 Script EDA sobre CSV sin orquestar limpieza (salida `reports/eda_raw/<task>/`) | [scripts/raw_eda.py](../scripts/raw_eda.py) — llama a `pre_processing.eda.run_eda` con columnas en snake_case |
| Comandos de reproducción (incl. Fase B) | [reproducir.txt](../reproducir.txt) — incluye línea comentada Fase B (`scripts/raw_eda.py`) |

*Pendiente de evolución del proyecto:* rellenar cifras medidas en `DATASET_INVENTORY.md` y generar carpetas `reports/eda_raw/...` por cada corpus/tabular exportado usado en el informe.

### Fase C — Preprocesamiento (alineado con el curso y el código)

| Paso | Detalle |
|------|---------|
| **C1. Diseño por tarea** | Estadificación: longitud de segmento, filtrado/remuestreo sobre **EEG crudo** si se avanza más allá de features agregadas; normalización **por sujeto vs por noche** (justificar). Apnea: misma política de ventana; definición clara de etiqueta vs anotación. |
| **C2. Implementación** | Uso de [`src/main.py`](../src/main.py) y [`pre_processing/`](../src/pre_processing/): limpieza → encoding → balanceo → escalado → dimensionalidad; EDA opcional sobre datos limpios. **Huecos vs PDF:** extracción rica de features (PSD, entropía, etc.) y **subject_id** en cada fila suelen requerir **nuevos scripts o notebooks**. |
| **C3. Congelar configuraciones** | JSON en [`config/datasets/`](../config/datasets/); salidas en `data/processed/` y reportes en `reports/` para citar en el informe. |

**Estado Fase C (v1.4):** el pipeline tabular (C2) ya está en `src/`; C1 y C3 quedan documentados y referenciados en la tabla siguiente. Los huecos del PDF (features espectrales en crudo, `subject_id` por fila) siguen siendo responsabilidad de notebooks/scripts de entrenamiento según el PRD.

| Entregable | Ubicación |
|------------|-----------|
| C1 Diseño por tarea (ventanas 30 s, política de normalización, límites del pipeline vs EEG crudo) | [docs/DISENO_PREPROCESAMIENTO.md](DISENO_PREPROCESAMIENTO.md) |
| C2 Implementación (referencia) | [`src/main.py`](../src/main.py), [`src/pre_processing/`](../src/pre_processing/) |
| C3 Índice de JSON congelados + comando tipo | [docs/CONFIG_DATASETS.md](CONFIG_DATASETS.md), JSON en [`config/datasets/`](../config/datasets/) (incl. `sleep_edf_expanded.json`) |
| Comandos de reproducción (incl. Fase C) | [reproducir.txt](../reproducir.txt) |

*Pendiente de evolución del proyecto:* ajustar cada JSON a los nombres reales de columnas de los CSV locales y registrar en el informe la política de escalado (global vs por sujeto cuando exista identificador).

### Fase D — Datos tras el preprocesamiento

| Paso | Detalle |
|------|---------|
| **D1. Segunda pasada de EDA** | Mismas ideas que B2 sobre la tabla **ya preprocesada**: balance tras resampling, escalas tras estandarización, número de features. |
| **D2. Figuras antes/después** | P. ej. distribución de clases, dimensionalidad, reducción de missingness — para la sección “Datos tras el preprocesamiento”. |

**Estado Fase D (v1.5):** segunda pasada de EDA sobre la tabla **post-pipeline** mediante `main.py --run-eda --run-eda-processed` (salida por defecto `reports/eda_processed/<task>/`); metodología y plantilla de informe enlazadas abajo.

| Entregable | Ubicación |
|------------|-----------|
| D1/D2 Metodología EDA procesado vs crudo / antes-después | [docs/EDA_POST_PREPROCESAMIENTO.md](EDA_POST_PREPROCESAMIENTO.md) |
| D2 Narrativa “Datos tras el preprocesamiento” (informe) | [docs/DATOS_TRAS_PREPROCESAMIENTO_PLANTILLA.md](DATOS_TRAS_PREPROCESAMIENTO_PLANTILLA.md) |
| Implementación (`--run-eda-processed`) | [`src/main.py`](../src/main.py) |
| Comandos de reproducción (incl. Fase D) | [reproducir.txt](../reproducir.txt) |

*Pendiente de evolución del proyecto:* generar pares de figuras/tablas antes-después por cada experimento citado en el informe.

### Fase E — Modelado (requisitos del PDF)

| Paso | Detalle |
|------|---------|
| **E1. Baselines de literatura** | 6–10 artículos y **≥3 baselines comparables**; tabla: paper, datos, canal, ventana, modelo, métricas. |
| **E2. Tres familias** | **SVM-RBF** (búsqueda de hiperparámetros solo en train); **Random Forest** + importancia de features; **Boosting** (XGBoost/GBM/AdaBoost) con mismo protocolo. Opcional: **HMM** (suavizado / transiciones). |
| **E3. Protocolos** | **Subject-wise k-fold** dentro de cada dataset; **cross-dataset** (entrenar en A, probar en B sin tunear en B). Guardar predicciones/matrices en **CSV** y figuras en **PNG**. |
| **E4. Métricas** | Apnea: accuracy, sensibilidad, especificidad, AUC-ROC. Sueño: accuracy, macro-F1, kappa; análisis por clase (**N1**). Estadística: media±std sobre folds, IC bootstrap, McNemar / AUC donde proceda. |

**Estado Fase E (v1.7):** orquestador `train_runner` con CV subject-wise y evaluación cross-dataset (hiperparámetros fijos en YAML); utilidades de métricas y artefactos. Pendiente típico del equipo: **E1** tabla de literatura en el informe, búsqueda extensa de hiperparámetros solo en train, **HMM**, bootstrap/IC.

| Entregable | Ubicación |
|------------|-----------|
| Runner CV + cross-dataset (SVM-RBF, RF, XGBoost) | [`src/modeling/train_runner.py`](../src/modeling/train_runner.py), [`scripts/run_phase_e_cv.py`](../scripts/run_phase_e_cv.py) |
| Esquema YAML + presets plug-and-play | [`config/experiment_train.example.yaml`](../config/experiment_train.example.yaml), [`config/experiment_train.schema.md`](../config/experiment_train.schema.md), [`config/experiments/README.md`](../config/experiments/README.md), [`docs/FASE_E_PLUG_AND_PLAY.md`](FASE_E_PLUG_AND_PLAY.md) |
| CV por sujeto | [`src/modeling/cv_split.py`](../src/modeling/cv_split.py) |
| Métricas E4 + McNemar | [`src/modeling/metrics.py`](../src/modeling/metrics.py) |
| Predicciones CSV + CM PNG | [`src/modeling/artifacts.py`](../src/modeling/artifacts.py) |
| `subject_unit_id` / unificación | [`src/modeling/subject_id.py`](../src/modeling/subject_id.py) |
| Guía + Colab | [docs/FASE_E_PREPARACION.md](FASE_E_PREPARACION.md) |
| Plantilla baselines (E1) | [docs/BASELINES_LITERATURA_PLANTILLA.md](BASELINES_LITERATURA_PLANTILLA.md) |

### Fase F — Análisis y discusión

| Paso | Detalle |
|------|---------|
| **F1. Preguntas guía del PDF** | Responder de forma explícita (preprocesamiento + features, desbalance/N1/apnea, accuracy vs macro-F1 vs kappa, matrices de confusión, mejor modelo in-dataset vs cross-dataset, fuga subject-wise vs diseños preliminares, solapamiento de IC, McNemar, importancia de features, ruido/cambio de base, SVM vs árboles, baseline más comparable, despliegue con EEG simple). |
| **F2. Comparación crítica** | Mejor resultado propio vs mejor baseline comparable; limitaciones honestas de la comparación. |

### Fase G — Propuesta de mejora (obligatoria; implementación = bono)

| Paso | Detalle |
|------|---------|
| **G1** | Teoría + diagrama + pseudocódigo; datos/código adicionales necesarios. |
| **G2** | Plan de evaluación justa frente a modelos ya probados. |
| **G3** | (Opcional) prototipo o ablación para puntos extra. |

### Fase H — Montaje del informe (estilo artículo)

| Paso | Detalle |
|------|---------|
| **H1** | Flujo tipo IMRaD: resumen (150–250 palabras) → intro → trabajos relacionados → datos y métodos → resultados → discusión → mejora → conclusiones → referencias IEEE → apéndice (comandos, IA). |
| **H2** | Checklist visual: resumen de datasets; EDA crudo; flujo de preprocesamiento; EDA procesado; tablas CV; matrices de confusión; cross-dataset; importancia de features; pruebas estadísticas. |
| **H3** | Control de calidad: cada cifra citada existe en CSV; Docker reproduce la tabla principal. |

### Fase I — Entregables del curso

Informe PDF, scripts de ejemplo, código en zip, README; autocontrol con la **rúbrica** (sección 8).

### Estado: implementado en repo vs pendiente

| Área | Estado |
|------|--------|
| Orquestación preprocesamiento tabular + EDA (`src/pre_processing`, `main.py`) | Implementado |
| Export WFDB → CSV 30 s (MIT-BIH, SHHS), `--source` ISRUC / St. Vincent / MIT-BIH / SHHS | Implementado |
| Preprocesamiento EEG específico (filtros, artefactos) + features espectrales avanzadas | Pendiente (típico en notebooks o nuevos módulos) |
| Fase E: runner YAML (CV + cross-dataset), SVM/RF/XGB, métricas y salidas | Implementado (v1.7) |
| HMM, tuning exhaustivo, bootstrap/IC, sección E1 en informe | Pendiente (equipo / curso) |
| Informe completo + baselines + propuesta de mejora | Pendiente |

---

## Referencias del curso (lista inicial del PDF)

Las referencias [1]–[12] del apéndice C de *Proyecto-1.pdf* (Ghimatgar et al., Ravan, Rechichi, Li, Gao, Satapathy, Wang, Chen, Pouliou, Xu, revisiones Heliyon/2024–2025) deben integrarse y ampliarse en la revisión de literatura del informe final.

---

*Documento generado para alinear el desarrollo del código y del informe con el enunciado oficial y con la implementación actual del repositorio. La sección 11 incorpora el roadmap; la Fase A queda documentada en v1.2 con artefactos en `docs/` y `config/`; la Fase B queda referenciada en v1.3 con artefactos en `docs/` y, para el CLI de EDA crudo, `scripts/raw_eda.py` + `reproducir.txt`; la Fase C queda referenciada en v1.4 con `DISENO_PREPROCESAMIENTO.md`, `CONFIG_DATASETS.md` y los JSON en `config/datasets/`; la Fase D queda referenciada en v1.5 con `EDA_POST_PREPROCESAMIENTO.md`, `DATOS_TRAS_PREPROCESAMIENTO_PLANTILLA.md` y `--run-eda-processed` en `main.py`; la Fase E queda referenciada en v1.7 con `modeling.train_runner`, `scripts/run_phase_e_cv.py`, `config/experiment_train.example.yaml`, `docs/FASE_E_PREPARACION.md` y `docs/BASELINES_LITERATURA_PLANTILLA.md`.*

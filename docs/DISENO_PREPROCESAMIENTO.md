# Diseño de preprocesamiento (Fase C — PRD)

Este documento cumple **C1** del [PRD §11](PRD.md): decisiones de método alineadas con [config/experiment_scope.yaml](../config/experiment_scope.yaml), el pipeline en [`src/main.py`](../src/main.py) y el informe final.

## C1.1 Unidad temporal y tareas

- **Ventana de 30 s:** acordada en `experiment_scope.yaml` (`tasks.*.epoch_seconds`) y con los exports WFDB→CSV (MIT-BIH, SHHS). Cualquier otro tamaño debe justificarse en el informe y actualizar el YAML si el equipo adopta un estándar nuevo.
- **Estadificación:** etiquetas objetivo en marco AASM (W, N1, N2, N3, REM) tras mapeo desde símbolos propios de cada corpus (`dataset_label_mapping` en el YAML).
- **Apnea / eventos:** la etiqueta depende del corpus (binario por ventana, tokens de evento, severidad en metadatos). No mezclar definiciones entre experimentos sin documentarlo; ver [ALCANCE_EXPERIMENTAL.md](ALCANCE_EXPERIMENTAL.md) §A1.2.

## C1.2 EEG y features más allá de la tabla

- **Filtrado, remuestreo y extracción rica (PSD, entropía, wavelets)** sobre EEG crudo no forman parte obligatoria de esta plantilla: el PRD las marca como trabajo típico en **notebooks o scripts nuevos**. El pipeline tabular asume **features ya agregadas por época** (p. ej. medias/desviaciones por canal en el CSV de ingesta/export).
- **Canal único EEG:** antes del modelado supervisado “solo EEG”, filtrar columnas según `eeg_single_channel` en el YAML y documentar el nombre exacto de la señal.

## C1.3 Normalización: por sujeto vs global

- **Ideal (informe / modelado):** si cada fila incluye `subject_id` (o equivalente), la estandarización **por sujeto** o particiones **subject-wise** evitan fuga entre train y test. Eso se implementa en el código de entrenamiento/validación, no solo en `scaling.py`.
- **Pipeline actual (`scaling.py`):** con `--scale-method` / `numeric_scaling` en JSON, el escalado opera sobre la **tabla completa** salvo exclusiones (`scale_exclude`). Si no hay identificador de sujeto en el CSV, documentar la limitación y priorizar añadir `subject_id` en ingesta o en un script previo.

## C1.4 Flujo implementado (C2)

Orden en `main.py`: limpieza → codificación → balanceo de clases → escalado → reducción de dimensionalidad → CSV procesado; EDA opcional sobre datos **limpios sin codificar** (`--run-eda`). Detalle de flags: [README.md](../README.md).

**Huecos explícitos (PRD):** `subject_id` por fila y features espectrales avanzadas suelen requerir **scripts o notebooks** adicionales.

## C1.5 Configuraciones congeladas (C3)

Especificaciones JSON por dataset/tarea: [CONFIG_DATASETS.md](CONFIG_DATASETS.md). Salidas esperadas: `data/processed/*.csv`, reportes bajo `reports/` (limpieza, encoding, etc.) para citar en el informe.

## Relación con Fase D

Tras obtener el CSV procesado, la segunda pasada de EDA (tabla final del pipeline) se documenta en [EDA_POST_PREPROCESAMIENTO.md](EDA_POST_PREPROCESAMIENTO.md) (`main.py --run-eda --run-eda-processed`).

## Relación con Fase E

Para particiones subject-wise, métricas y guardado de resultados antes de implementar el entrenamiento, ver [FASE_E_PREPARACION.md](FASE_E_PREPARACION.md) y el paquete `src/modeling/`.

---

*Mantener sincronizado con `experiment_scope.yaml` y con las rutas reales de CSV del equipo.*

# EDA tras el preprocesamiento (Fase D — PRD)

Cumple **D1** y guía **D2** del [PRD §11](PRD.md): segunda pasada de análisis exploratorio sobre la tabla **ya transformada** por el pipeline (codificación, balanceo opcional, escalado, reducción de dimensionalidad), no solo sobre la tabla limpia previa a one-hot.

## D1. Qué contrastar con la Fase B

| Aspecto | Fase B (crudo / export) | Fase D (procesado) |
|--------|-------------------------|---------------------|
| Entrada típica | CSV de export o ingesta; [`scripts/raw_eda.py`](../scripts/raw_eda.py) | Misma ruta lógica tras [`src/main.py`](../src/main.py) con `--run-eda` y **`--run-eda-processed`** |
| Carpeta de reportes por defecto | `reports/eda_raw/<task>/` | `reports/eda_processed/<task>/` |
| Contenido | Missingness y clases “como llegan” | Distribución de clases tras SMOTE/under/over; escalas homogéneas si hubo estandarización; **número de columnas** tras one-hot o PCA |

El módulo [`pre_processing/eda.py`](../src/pre_processing/eda.py) genera el mismo tipo de artefactos (perfiles, correlaciones, figuras, `eda_summary.md`); cambia el **dataframe** de entrada.

## D2. Figuras y tablas “antes / después”

Para el informe, conviene **emparejar** artefactos de:

- **Antes:** Fase B (`reports/eda_raw/...`) o, si solo se dispone del pipeline, EDA sobre datos solo limpios (`--run-eda` sin `--run-eda-processed` → `reports/eda/...`).
- **Después:** Fase D (`--run-eda --run-eda-processed` → `reports/eda_processed/...`).

Elementos útiles para la sección “Datos tras el preprocesamiento”:

- `fig_target_distribution.png` — balance de clases antes vs después de remuestreo.
- Número de columnas en `01_dataset_profile.csv` o en el CSV de salida — dimensionalidad tras one-hot / PCA.
- Missingness reducido — comparar perfiles si la limpieza imputó o eliminó columnas.

Plantilla narrativa: [DATOS_TRAS_PREPROCESAMIENTO_PLANTILLA.md](DATOS_TRAS_PREPROCESAMIENTO_PLANTILLA.md).

## Comando (referencia)

Ver [reproducir.txt](../reproducir.txt) (línea Fase D). Ejemplo:

```bash
python src/main.py --input data/processed/in.csv --output data/processed/out.csv --encoding-spec config/datasets/mitbih_sleep_stages.json --target-col sleep_stage --task mitbih_d --run-eda --run-eda-processed --write-cleaning-report
```

Si se omite `--eda-outdir`, los informes de Fase D van a `reports/eda_processed/<task>/`.

---

*Las métricas finales del modelo siguen protocolo train/test (p. ej. subject-wise); este EDA es descriptivo sobre la tabla que alimentará o refleja el entrenamiento.*

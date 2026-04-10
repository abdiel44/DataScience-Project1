# Alcance experimental (Fase A — completado como línea base)

Este documento **fija el contrato** entre el equipo, el [PRD](PRD.md) y [Proyecto-1.pdf](Proyecto-1.pdf): qué se estudia, con qué etiquetas, qué ventanas y qué política de EEG monocanal. Las decisiones finales del informe deben **coincidir** con [config/experiment_scope.yaml](../config/experiment_scope.yaml) o actualizar ambos a la vez.

## A1.1 Tarea 1: Estadificación del sueño (multicategoría)

- **Marco teórico:** etapas según AASM (resumen en apéndice del PDF): W, N1, N2, N3, REM.
- **Unidad de análisis:** épocas de **30 s**, alineadas con el enunciado y con los exports WFDB→CSV del repo (MIT-BIH, SHHS).
- **Mapeo por corpus:** cada base puede usar símbolos distintos (`W`, `1`, …). La tabla de mapeo **corpus → {W,N1,N2,N3,REM}** se mantiene en `config/experiment_scope.yaml` bajo `dataset_label_mapping` y se amplía en el informe con cualquier excepción justificada.

## A1.2 Tarea 2: Apnea / eventos respiratorios

- **Definición operativa:** depende del corpus (binario por ventana, lista de eventos en anotaciones, o severidad global en metadatos). En este repositorio:
  - **MIT-BIH:** filas de eventos con `event_tokens` (tokens H, HA, OA, …) y filas de sueño con `sleep_stage` desde el anotador `st`.
  - **SHHS:** eventos en export separado (`resp`, `arou`) con ventana 30 s alineada al centro del evento.
  - **St. Vincent / ISRUC:** según columnas derivadas de los ingest actuales (ver YAML).
- El informe debe **explicitar** qué se usa como etiqueta de apnea en cada experimento (no mezclar definiciones sin aviso).

## A1.3 EEG monocanal

El proyecto exige **un solo canal EEG** (frontal o central estándar). Los CSV generados por el pipeline pueden contener **varias señales PSG**; antes del modelado supervisado el equipo debe:

1. Seleccionar el **nombre exacto** de columna o señal por dataset (sugerencias iniciales en `experiment_scope.yaml` → `eeg_single_channel.recommended_signals`).
2. Descartar el resto de columnas de señal o no usarlas como features si el informe reclama “solo EEG”.
3. Documentar la elección en métodos (reproducibilidad).

## A2. Reproducibilidad (referencia cruzada)

- Comandos y entorno: [**reproducir.txt**](../reproducir.txt) en la raíz del repo.
- Semillas globales: [**config/reproducibility.env.example**](../config/reproducibility.env.example) (copiar a `.env` si se usa dotenv en scripts de entrenamiento).
- Versiones de librerías: `requirements.txt`; para congelado estricto opcional ver `requirements-pinned.txt` si existe.

## A3. Uso responsable de IA

Plantilla breve para el informe y bitácora: [**USO_IA_PLANTILLA.md**](USO_IA_PLANTILLA.md).

---

*Actualizar este archivo al cerrar decisiones definitivas (p. ej. canal EEG único por dataset tras pilotos).*

Para el diseño de preprocesamiento tabular y política de normalización (Fase C), ver también [DISENO_PREPROCESAMIENTO.md](DISENO_PREPROCESAMIENTO.md).

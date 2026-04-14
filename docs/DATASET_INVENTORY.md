# Inventario de datasets (Fase B — B1)

Tabla de referencia para el informe. Las cifras siguientes corresponden a la copia local actual bajo `data/raw` en esta máquina.

| Dataset | Fuente / acceso | Registros / sujetos (local actual) | Duración / unidad | fs (notas) | Canales en export del repo | Anotación | Documentación |
|---------|-----------------|-------------------------------------|-------------------|------------|-----------------------------|-----------|---------------|
| Sleep-EDF Expanded | PhysioNet / descarga local | **197** PSG/hypnogram EDF emparejados | Noche completa | 100 Hz en el PSG inspeccionado | Resumen vía `mne` + `sleep_stage` dominante por registro | Hipnograma EDF, colapsado a W/N1/N2/N3/REM | PhysioNet sleep-edf-database |
| MIT-BIH PSG (SLPDB) | PhysioNet | **18** registros en `RECORDS` | ~2 h c/u (ej.) | 250 Hz (ej. slp01a) | CSV época 30 s: varias señales PSG + `sleep_stage` | `st`: estadios + eventos en aux | PhysioNet slpdb |
| SHHS | PhysioNet muestra | **1** registro en `RECORDS` | Larga | En .hea del registro | CSV época 30 s + eventos resp/arou | `hypn`, `resp`, `arou` | SHHS docs |
| St. Vincent Apnea | PhysioNet | **25** archivos `ucddb*_stage.txt` | Resumen por noche | N/A en ingesta actual | Resumen por noche (`stage_mode`, fracciones) | Enteros 0–5 en `stage.txt` | Dataset page |
| ISRUC-Sleep | Carpeta local | **36,162** CSV de segmento | Ventana corta por CSV | Variable por segmento | Agregados mean/std por canal + `event_group` / `sleep_stage` | Ruta + nombre archivo | ISRUC doc |

**EEG monocanal (requisito del proyecto):** tras elegir una señal por corpus, añadir una fila al informe o columna “Canal usado para ML” en esta tabla.

---

*Generado para cumplir PRD Fase B1; mantener sincronizado con `config/experiment_scope.yaml`.*

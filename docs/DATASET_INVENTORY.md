# Inventario de datasets (Fase B — B1)

Tabla de referencia para el informe. **Actualizar** filas con cifras medidas en tu copia local de `data/raw` (los recuentos cambian si añades o quitas registros).

| Dataset | Fuente / acceso | Registros / sujetos (típico) | Duración / unidad | fs (notas) | Canales en export del repo | Anotación | Documentación |
|---------|-----------------|------------------------------|-------------------|------------|-----------------------------|-----------|---------------|
| Sleep-EDF Expanded | PhysioNet / descarga local | Varios PSG nocturnos | Noche completa | Depende del registro (p. ej. 100 Hz) | Resumen vía `mne` opcional; EDF + hypnogram | Etapas sueño; hipnograma | PhysioNet sleep-edf-database |
| MIT-BIH PSG (SLPDB) | PhysioNet | 18 registros en `RECORDS` | ~2 h c/u (ej.) | 250 Hz (ej. slp01a) | CSV época 30 s: varias señales PSG + `sleep_stage` | `st`: estadios + eventos en aux | PhysioNet slpdb |
| SHHS | PhysioNet muestra | Depende de `RECORDS` (ej. 1+) | Larga | En .hea (ej. 1 Hz marcos) | CSV época 30 s + eventos resp/arou | `hypn`, `resp`, `arou` | SHHS docs |
| St. Vincent Apnea | PhysioNet | `ucddb*_stage.txt` (ej. 25+) | Por época en txt | N/A en ingesta actual | Resumen por noche (`stage_mode`, fracciones) | Enteros 0–5 en stage.txt | Dataset page |
| ISRUC-Sleep | Carpeta local | Miles de CSV segmento | Ventana corta por CSV | Variable por segmento | Agregados mean/std por canal + `event_group` / `sleep_stage` | Ruta + nombre archivo | ISRUC doc |

**EEG monocanal (requisito del proyecto):** tras elegir una señal por corpus, añadir una fila al informe o columna “Canal usado para ML” en esta tabla.

---

*Generado para cumplir PRD Fase B1; mantener sincronizado con `config/experiment_scope.yaml`.*

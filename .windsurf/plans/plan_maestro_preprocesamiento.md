# Plan Maestro de Preprocesamiento y Evaluacion

Este documento consolida el diseno metodologico del proyecto de `sleep staging` y `deteccion de apnea` con `EEG monocanal`. Su funcion es integrar en un solo flujo los requerimientos del curso y los planes detallados ya definidos para EDA, limpieza, balance de clases y escalado.

## Documentos de referencia y trazabilidad

### Requerimientos base

- [proyecto_1_requerimientos.md](./proyecto_1_requerimientos.md)
- [Resumen.md](./Resumen.md)

### Planes detallados ya aprobados

- [eda_eeg_single-channel_1b1e9149.plan.md](./eda_eeg_single-channel_1b1e9149.plan.md)
- [limpieza_de_datos_eeg_ad854048.plan.md](./limpieza_de_datos_eeg_ad854048.plan.md)
- [equilibrio_de_clases_topic_11_effd4067.plan.md](./equilibrio_de_clases_topic_11_effd4067.plan.md)
- [módulo_scaling_topic_5_a6ec59c8.plan.md](./módulo_scaling_topic_5_a6ec59c8.plan.md)

## Principio rector

El proyecto se ejecutara en dos niveles:

1. **Curacion por dataset**

- cada base se audita, limpia, segmenta, homologa y documenta por separado;
- de cada base principal sale un **dataset curado o fijo**;
- ese dataset fijo no se redefine en cada comparacion de modelos.

2. **Experimentacion sobre datasets curados**

- sobre la version fija se ejecutan `features`, `splits`, `SVM-RBF`, `Random Forest` y `Boosting` en un flujo secuencial;
- primero se valida la familia metodologica en `apnea`, incluyendo resultados por ventana y por paciente;
- despues se reutiliza la misma base metodologica para `staging`, mediante adaptacion y reentrenamiento supervisado;
- luego se hacen comparaciones `within-dataset`, `cross-dataset`, matrices de confusion, metricas por paciente y analisis de generalizacion;
- la comparacion principal sera `baseline minimo de preprocesamiento` vs `pipeline final de preprocesamiento`, no `processed vs raw bruto`.

Flujo obligatorio de alto nivel:

`raw dataset -> auditoria -> dataset curado/fijo -> features -> entrenamiento apnea -> evaluacion clinica y por paciente -> seleccion del pipeline/modelo mas solido -> adaptacion y reentrenamiento para staging -> evaluacion staging -> cross-dataset -> comparacion con literatura`

## Alcance y criterio de cierre

El plan se considera completo solo cuando incluya:

- validacion `subject-wise` sin fuga de informacion;
- experimentos `cross-dataset` en las bases habilitadas;
- comparacion con `6-10` referencias cientificas y al menos `3` baselines bien documentados;
- analisis explicito de sobreajuste y cambio de dominio;
- una fase `apnea` resuelta antes de la fase `staging`;
- una propuesta de mejora metodologica sobre el mejor baseline propio.

## Politica de datasets

### Bases principales de trabajo

- `Sleep-EDF`: base principal para `staging`.
- `ISRUC-Sleep`: base complementaria para `staging`.
- `MIT-BIH PSG`: base principal para `apnea`.

### Bases de generalizacion externa

- `SHHS`: base amplia para `generalizacion externa` en apnea; no se tunea ahi.
- `St. Vincent's`: base clinica adicional para evaluacion externa; no se tunea ahi y solo entra si se valida el canal EEG equivalente.

### Alcance controlado

- aunque algunos requerimientos mencionan usos adicionales potenciales para `Sleep-EDF`, `MIT-BIH` o `SHHS`, la **secuencia minima obligatoria** de este proyecto queda fijada en:
  - `MIT-BIH PSG` + `apnea` como fase 1
  - `Sleep-EDF` + `staging` como fase 2
  - `ISRUC-Sleep` + `staging` como fase 2
- cualquier extension extra se considera opcional y no bloquea la entrega minima.

### Estado actual de la copia local

- `SHHS` sigue en modo `audit-only` mientras la copia local continue incompleta.
- `St. Vincent's` sigue en modo `audit-only` mientras no se confirme el canal EEG estandar o equivalente.

## Dataset curado o fijo por base principal

Cada dataset principal debe producir una version curada con estas propiedades:

- un solo canal EEG definido y documentado;
- ventanas consistentes de `30 s`;
- etiquetas homologadas;
- `subject_id`, `record_id`, `session_id` y `fold_group` definidos;
- `source_file`, `window_start_s`, `window_end_s`, `sampling_rate_hz`, `eeg_channel` definidos;
- archivos `CSV` reproducibles que luego se reutilizan en todos los experimentos.

### Sleep-EDF

- PSG e hipnograma sincronizados.
- exclusion de `?` y `Movement`.
- fusion `N4 -> N3`.
- canal retenido: `EEG Fpz-Cz`.
- recorte del subconjunto `sleep-cassette` al bloque nocturno.

### ISRUC-Sleep

- la copia local se trata como dataset ya segmentado.
- validacion del nombre de archivo.
- mapeo `Stagew/Stagen1/Stagen2/Stagen3/Stager -> W/N1/N2/N3/REM`.
- seleccion de canal central.

### MIT-BIH PSG

- sincronizacion de senal y anotaciones.
- retencion exclusiva de `EEG (C3-O1)` o `EEG (C4-A1)`.
- descarte de registros cuyo unico EEG sea `O2-A1`.
- construccion de ventanas `apnea/no_apnea`.

## 1. Auditoria de datasets y EDA

Esta etapa tiene dos subniveles y ambos son obligatorios.

### 1A. Auditoria de viabilidad por dataset

Objetivo:

- decidir que datasets entran al desarrollo, cuales sirven como prueba externa y cuales quedan bloqueados.

Tareas:

- inventariar sujetos, registros, canal EEG usable, frecuencia de muestreo y origen de etiquetas;
- medir balance de clases, duracion de registros y porcentaje de vigilia;
- crear matriz de viabilidad por dataset;
- documentar por que `SHHS` y `St. Vincent's` estan habilitados, limitados o bloqueados.

Salidas:

- `summary.csv`
- `records.csv`
- `labels.csv`
- `viability.csv`
- `label_distribution.png`
- `subject_distribution.png`

### 1B. EDA reproducible sobre datos curados

Referencia detallada: [eda_eeg_single-channel_1b1e9149.plan.md](./eda_eeg_single-channel_1b1e9149.plan.md)

Objetivo:

- caracterizar estructura, calidad, distribucion y relaciones entre variables del dataset ya limpio o ya curado;
- producir artefactos reutilizables para el informe final y para justificar decisiones de limpieza, balanceo, escalado y modelado.

Salidas minimas por dataset:

- `01_dataset_profile.csv`
- `02_descriptive_numeric.csv`
- `03_descriptive_categorical.csv`
- `04_correlations.csv`
- `fig_hist_<feature>.png`
- `fig_box_<feature>.png`
- `fig_corr_heatmap.png`
- `fig_target_distribution.png`
- `eda_summary.md`

Nota metodologica:

- la auditoria de viabilidad puede ejecutarse sobre datos de inventario;
- el EDA analitico principal debe ejecutarse sobre el dataframe ya limpio o sobre el dataset curado, y su orden debe quedar documentado en el reporte.
- este EDA aplica antes de la fase `apnea` y antes de la fase `staging`; cambia el orden estrategico de entrenamiento, no la necesidad del subplan.

## 2. Curacion, limpieza y preprocesamiento de senal

Referencia detallada: [limpieza_de_datos_eeg_ad854048.plan.md](./limpieza_de_datos_eeg_ad854048.plan.md)

### Objetivo

Eliminar inconsistencias, controlar ruido y artefactos, y convertir cada fuente a ventanas validas de analisis.

### Reglas minimas de limpieza estructural

- tratar `missing`, `duplicados`, inconsistencias de tipos y normalizacion de texto;
- definir politica explicita para `target` faltante;
- permitir deduplicacion por clave semantica cuando corresponda;
- reportar coerciones numericas, imputaciones, columnas eliminadas y outliers tratados;
- generar un `cleaning_summary.md` y, cuando aplique, `cleaning_log.csv`.

### Reglas minimas de senal EEG

- documentar el canal retenido por dataset;
- segmentar en ventanas consistentes de `30 s`;
- definir y documentar filtrado y/o remuestreo apropiado para el canal EEG seleccionado;
- documentar la normalizacion de amplitud por `sujeto`, por `registro`, o justificar formalmente por que no se aplica;
- describir como se detectan, excluyen o modelan artefactos/ruido.

### Politica por dataset

- `Sleep-EDF`: emparejar PSG + hipnograma, quitar `?` y `Movement`, mapear `N4 -> N3`, recortar `sleep-cassette` y conservar `EEG Fpz-Cz`.
- `ISRUC-Sleep`: validar nombres de archivo, mapear stages y seleccionar canal central.
- `MIT-BIH PSG`: descartar registros sin `C3-O1` o `C4-A1` y derivar apnea/no-apnea desde `.st`.
- `SHHS`: auditar estructura y bloquear entrenamiento mientras la copia local siga incompleta.
- `St. Vincent's`: auditar y bloquear mientras el canal EEG no este validado.

### Outliers y ruido

- la definicion minima de outlier tabular sera `Tukey 1.5 x IQR`;
- la accion por defecto sera conservadora: `winsorize`, transformacion o exclusion justificada, no borrado masivo sin evidencia;
- el tratamiento de outliers debe ser coherente con los boxplots y tablas del EDA;
- los artefactos de senal deben quedar documentados aunque no se apliquen tecnicas avanzadas de remocion.

### Salidas

- `windows_clean.csv`
- `cleaning_summary.md`
- `cleaning_log.csv` cuando aplique
- `signal_preprocessing_plan.md`

Nota de integracion:

- este subplan de limpieza aplica tanto a la fase `apnea` como a la fase `staging`; la secuencia nueva no reemplaza este bloque, solo redefine que tarea se entrena primero.

## 3. Codificacion y preparacion de variables

### Objetivo

Normalizar la representacion de ventanas para impedir fugas y comparar datasets con una sola interfaz.

### Tareas

- usar ventanas de `30 s`;
- generar columnas canonicas:
  - `dataset`, `task`, `subject_id`, `record_id`, `session_id`, `window_id`
  - `window_start_s`, `window_end_s`, `sampling_rate_hz`, `sampling_rate_target_hz`
  - `eeg_channel`, `label_raw`, `label_final`, `label_code`, `source_file`, `fold_group`
- `staging`: etiquetas finales `W/N1/N2/N3/REM`;
- `apnea`: `apnea` / `no_apnea`;
- asegurar trazabilidad entre ventana, sujeto y registro para futuros `GroupKFold` y metricas por paciente.

### Salidas

- `windows_prepared.csv`
- `windows_summary.csv`

## 4. Balance de clases

Referencia detallada: [equilibrio_de_clases_topic_11_effd4067.plan.md](./equilibrio_de_clases_topic_11_effd4067.plan.md)

### Objetivo

Manejar el desbalance sin introducir fuga de informacion y sin alterar el dataset curado base.

### Reglas

- el balanceo **nunca** se hace antes del split `subject-wise`;
- todo remuestreo o `class_weight` se ajusta solo con el `train-fold`;
- el conjunto de validacion o prueba conserva la distribucion real del problema;
- se pueden comparar `class_weight`, `random_under`, `random_over` y `SMOTE` segun la tarea y el modelo;
- si se usa `SMOTE`, debe ejecutarse solo sobre espacio numerico ya preparado y solo dentro de entrenamiento.

### Salidas

- `class_balance_summary.md`
- `class_balance_before_after.csv`
- `class_weight_plan.csv`

Nota de integracion:

- el balanceo sigue aplicando dentro del `train-fold` en la fase `apnea` y en la fase `staging`; lo que cambia es que el primer ciclo completo de comparacion se hace en apnea.

## 5. Escalado y transformacion numerica

Referencia detallada: [módulo_scaling_topic_5_a6ec59c8.plan.md](./módulo_scaling_topic_5_a6ec59c8.plan.md)

### Objetivo

Definir una politica de transformacion numerica compatible con validacion `subject-wise`, con clases desbalanceadas y con el tipo de modelo.

### Aclaracion metodologica

- `remuestreo de senal` y `filtrado` pertenecen a la etapa de preprocesamiento de senal, no a la etapa de escalado;
- esta etapa se refiere a `scaling` y transformaciones sobre **features** ya construidas;
- el escalado ocurre despues de codificacion y, cuando aplique, despues del balanceo dentro del `train-fold`.

### Tareas

- aplicar `log1p` a features espectrales no negativas cuando corresponda;
- usar `StandardScaler` para `SVM-RBF` y para modelos de `Boosting` sensibles a escala cuando sea beneficioso;
- evaluar `minmax` solo si la distribucion y los outliers lo justifican;
- mantener `Random Forest` sin dependencia obligatoria del escalado;
- excluir `target`, identificadores y columnas que no deban escalarse.

### Salidas

- `baseline_scaling_plan.csv`
- `final_scaling_plan.csv`
- `scaling_summary.md`

Nota de integracion:

- el escalado sigue aplicando dentro del `train-fold` en ambas fases; la diferencia es que la seleccion inicial de politicas y familias de modelo se decide primero con apnea.

## 6. Ingenieria de caracteristicas y reduccion de dimensiones

### Objetivo

Comparar un baseline minimo contra un pipeline final mas rico.

### Baseline minimo

- media, desviacion estandar, min, max, RMS, IQR;
- `zero crossing rate`;
- potencias absolutas por banda.

### Pipeline final

- todo el baseline;
- `skew`, `kurtosis`, `line length`;
- parametros de `Hjorth`;
- `spectral entropy`;
- potencias relativas y razones entre bandas;
- reduccion dentro de cada fold:
  - `VarianceThreshold`
  - filtro por correlacion `|r| > 0.95`
  - `SelectKBest(mutual_info_classif, k=40)`
  - `PCA 95%` como rama para `SVM`

### Regla critica

- toda seleccion, reduccion o calibracion se ajusta solo con el `train-fold`;
- el dataset curado fijo no se reescribe por cambiar el conjunto de features.

### Salidas

- `baseline_features.csv`
- `final_features.csv`
- `feature_catalog.csv`

## 7. Particion de datos y protocolos de validacion

### Objetivo

Evaluar correctamente sin fuga de informacion.

### Reglas

- toda particion debe ser `subject-wise`;
- nunca usar `leave-one-window-out`;
- toda normalizacion, seleccion de features, balanceo o calibracion debe ajustarse solo con el fold de entrenamiento;
- fijar `seed` y registrar configuracion exacta de cada corrida.

### Protocolos internos

- `Sleep-EDF`: `GroupKFold(5)` para `staging`.
- `ISRUC-Sleep`: `GroupKFold(5)` para `staging`.
- `MIT-BIH PSG`: `Leave-One-Subject-Out` para `apnea`.

### Cross-dataset minimo

- `Sleep-EDF -> ISRUC`
- `ISRUC -> Sleep-EDF`
- `MIT-BIH -> SHHS` solo cuando `SHHS` este habilitado
- `MIT-BIH -> St. Vincent's` solo cuando se valide EEG

### Salidas

- `partition_assignments.csv`
- `partition_summary.csv`
- `cross_dataset_protocols.csv`

## 8. Modelos, tuning y alcance obligatorio

### Modelos obligatorios

- `SVM-RBF`
- `Random Forest`
- `Boosting`

### Parametros minimos a registrar

- `SVM-RBF`: `C`, `gamma`, `class_weight`, estrategia multiclase;
- `Random Forest`: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`;
- `Boosting`: variante elegida (`AdaBoost`, `Gradient Boosting` o equivalente), hiperparametros clave y razon de seleccion.

### Logica secuencial de entrenamiento

- las tres familias se comparan primero en la tarea de `apnea`;
- la fase inicial obligatoria usa `MIT-BIH PSG` para `apnea/no_apnea` por ventana;
- esa misma fase debe incluir agregacion por paciente para estimar `AHI` o un criterio equivalente documentado, y clasificacion de severidad por paciente;
- si los datasets o anotaciones lo permiten, se pueden explorar biomarcadores adicionales de apnea como extension opcional;
- una vez identificada la familia o familias con mejor rendimiento, robustez e interpretabilidad en apnea, se reutiliza su base metodologica para `staging`.

### Criterio de transferencia metodologica

- la reutilizacion entre tareas se entiende como `misma familia de modelos + mismo pipeline metodologico`;
- se conservan lineamientos de `features`, politicas de balanceo, escalado, seleccion y tuning que hayan sido validados en apnea;
- para `staging` se realiza un nuevo ajuste supervisado sobre sus datasets, no una continuacion literal del entrenamiento previo;
- no se afirmara transferencia de pesos o del objeto entrenado para `RF`, `SVM` o `Boosting`, sino adaptacion y reentrenamiento con la misma base metodologica.

### Aclaracion sobre HMM

- `HMM` queda como linea opcional o exploratoria;
- no forma parte de las `9 configuraciones principales` que bloquean el cierre del plan;
- si se implementa, debe reportarse como experimento adicional y no sustituye a `SVM`, `RF` y `Boosting`.

## Matriz de experimentos minima

### Fase 1. Apnea

| Tarea    | Dataset         | Rol             | Modelos                 | Validacion interna  | Validacion externa            |
| -------- | --------------- | --------------- | ----------------------- | ------------------- | ----------------------------- |
| `apnea`  | `MIT-BIH`       | principal       | `SVM`, `RF`, `Boosting` | `LOSO` subject-wise | probar en `SHHS`              |
| `apnea`  | `SHHS`          | externo         | no se tunea ahi         | no aplica al inicio | test externo                  |
| `apnea`  | `St. Vincent's` | clinico externo | no se tunea ahi         | no aplica al inicio | test externo si se valida EEG |

### Fase 2. Staging

| Tarea     | Dataset       | Rol            | Modelos reutilizados metodologicamente | Validacion interna        | Validacion externa    |
| --------- | ------------- | -------------- | -------------------------------------- | ------------------------- | --------------------- |
| `staging` | `Sleep-EDF`   | principal      | `SVM`, `RF`, `Boosting`                | `GroupKFold` subject-wise | probar en `ISRUC`     |
| `staging` | `ISRUC-Sleep` | complementario | `SVM`, `RF`, `Boosting`                | `GroupKFold` subject-wise | probar en `Sleep-EDF` |

Nota:

- la fase `staging` depende del aprendizaje obtenido en la fase `apnea`, pero ese aprendizaje se traduce en seleccion metodologica y no en transferencia literal del modelo ya ajustado.

## Bloques experimentales obligatorios

### Nueve configuraciones principales

- `MIT-BIH + apnea + SVM`
- `MIT-BIH + apnea + RF`
- `MIT-BIH + apnea + Boosting`
- `Sleep-EDF + staging + SVM`
- `Sleep-EDF + staging + RF`
- `Sleep-EDF + staging + Boosting`
- `ISRUC + staging + SVM`
- `ISRUC + staging + RF`
- `ISRUC + staging + Boosting`

### Aclaracion

- si son `9 configuraciones principales`;
- las `3` primeras pertenecen a la fase obligatoria de `apnea`;
- las `6` restantes pertenecen a la fase de `staging` derivada de la transferencia metodologica;
- no son `9 entrenamientos reales`, porque cada configuracion se ejecuta con folds;
- por eso el numero real de entrenamientos es mayor:
  - `MIT-BIH`: `3 modelos x LOSO`
  - `Sleep-EDF`: `3 modelos x 5 kfolds`
  - `ISRUC`: `3 modelos x 5 kfolds`

## 9. Evaluacion obligatoria

### Metricas por ventana o epoch

- `apnea`: `accuracy`, `sensibilidad`, `especificidad`, `AUC-ROC`.
- `staging`: `accuracy`, `macro-F1`, `Cohen's kappa`, `N1 F1`.

### Metricas por paciente para apnea

- agregar predicciones por sujeto o registro para estimar resumen clinico;
- calcular `AHI` estimado o un criterio equivalente documentado;
- calcular exactitud de severidad por paciente a partir del criterio `AHI` adoptado;
- reportar sensibilidad y especificidad para los puntos de corte por paciente definidos por el equipo.

### Secuencia de evaluacion

- primero se cierra la evaluacion de `apnea/no_apnea` por ventana;
- despues se agregan resultados por paciente para `AHI`, severidad y cualquier biomarcador adicional soportado por el dataset;
- solo despues de identificar las familias metodologicas mas solidas en apnea se abre la fase de `staging`;
- en `staging` se reaplican las mismas reglas de validacion, balanceo, escalado y comparacion `within/cross-dataset`.

### Que debe tener cada bloque experimental

- `baseline minimo`
- `pipeline final`
- metricas por fold
- media y desviacion estandar
- matriz de confusion agregada
- metricas de entrenamiento y prueba por fold
- resultados `within-dataset`
- resultados `cross-dataset` cuando el protocolo este habilitado
- tiempos de entrenamiento e inferencia
- hiperparametros y `seed`

### Salidas

- `*_within_metrics.csv`
- `*_cross_metrics.csv`
- `*_within_confusion.csv`
- `*_cross_confusion.csv`
- `*_importance.csv`
- `patient_level_metrics.csv`
- `ahi_patient_summary.csv`
- `experiment_registry.csv`

## 10. Comparacion con literatura y baselines publicados

### Objetivo

Situar los resultados propios frente a trabajos reconocidos y justificar la mejora metodologica propuesta.

### Minimos obligatorios

- revisar `6-10` referencias cientificas relevantes;
- comparar contra al menos `3` baselines bien documentados;
- explicitar diferencias de dataset, canal, etiquetado, protocolo y metricas;
- proponer una mejora metodologica concreta a partir de los hallazgos del proyecto.

### Salidas

- `literature_matrix.csv`
- `baseline_comparison.md`
- `improvement_proposal.md`

## 11. Reproducibilidad y entregables

### Requisitos obligatorios

- mantener entorno reproducible con `Docker`;
- guardar todo resultado tabular en `CSV`;
- guardar toda figura final en `PNG`;
- registrar hiperparametros, `seeds`, tiempos y particiones usadas;
- incluir un archivo `.txt` con comandos ejemplo de ejecucion;
- mantener `README.md` con dependencias, estructura, datasets y pasos de reproduccion.

### Salidas

- `Dockerfile`
- `docker-compose.yml`
- `reports/run_commands.txt`
- `README.md`
- `experiment_registry.csv`

## Criterio de overfitting

El sobreajuste no se evaluara solo con el rendimiento dentro del mismo dataset.

### Senales que deben revisarse

- diferencia entre metricas de `train` y `test` por fold;
- alta varianza entre folds;
- caida fuerte del rendimiento al pasar de `within-dataset` a `cross-dataset`;
- mejora aparente del pipeline final en validacion interna que no se sostiene en validacion externa.

### Regla de interpretacion

- si falla en `within-dataset` y en `cross-dataset`, el modelo generaliza mal;
- si funciona bien dentro del dataset pero cae mucho fuera, esta sobreajustado al dominio fuente;
- si mantiene desempeno razonablemente estable entre folds internos y en bases externas no vistas, es mas robusto.

## Orden recomendado de ejecucion

1. Auditar `Sleep-EDF`, `ISRUC`, `MIT-BIH`, `SHHS` y `St. Vincent's`.
2. Curar y limpiar `MIT-BIH`.
3. Curar y limpiar `Sleep-EDF`.
4. Curar y limpiar `ISRUC`.
5. Congelar los tres datasets curados.
6. Ejecutar EDA reproducible por dataset curado.
7. Construir `baseline_features` y `final_features`.
8. Definir splits `subject-wise`.
9. Correr `balance -> scale -> select -> model` dentro de cada `train-fold`.
10. Ejecutar `SVM`, `RF` y `Boosting` sobre `MIT-BIH` para `apnea/no_apnea`.
11. Agregar resultados por paciente para `AHI`, severidad y biomarcadores opcionales si existen.
12. Seleccionar la familia o familias con mejor desempeno, robustez, interpretabilidad y estabilidad.
13. Trasladar esa base metodologica a `Sleep-EDF` y `ISRUC`.
14. Reentrenar y adaptar `SVM`, `RF` y `Boosting` para `staging`.
15. Ejecutar `within-dataset` y `cross-dataset` en las bases habilitadas.
16. Comparar con literatura y baselines publicados.
17. Redactar limitaciones, sobreajuste y mejora metodologica.

## Resumen operativo

- `MIT-BIH` es la base principal de `apnea` y define la primera fase obligatoria del proyecto.
- `Sleep-EDF` e `ISRUC` son las bases principales de `staging` y se abordan despues de cerrar la fase de apnea.
- `SHHS` y `St. Vincent's` sirven como bases de generalizacion externa cuando sean utilizables.
- el maestro integra y remite explicitamente a:
  - [eda_eeg_single-channel_1b1e9149.plan.md](./eda_eeg_single-channel_1b1e9149.plan.md)
  - [limpieza_de_datos_eeg_ad854048.plan.md](./limpieza_de_datos_eeg_ad854048.plan.md)
  - [equilibrio_de_clases_topic_11_effd4067.plan.md](./equilibrio_de_clases_topic_11_effd4067.plan.md)
  - [módulo_scaling_topic_5_a6ec59c8.plan.md](./módulo_scaling_topic_5_a6ec59c8.plan.md)
- esos cuatro subplanes aplican antes de `apnea` y antes de `staging`; cambia el orden estrategico de entrenamiento, no el reemplazo de los subplanes.
- la transferencia entre tareas se interpreta como `misma familia de modelos + mismo pipeline metodologico`, no como reutilizacion literal del objeto entrenado.
- la comparacion principal es `baseline preprocessing vs final preprocessing`, no `processed vs raw bruto`.
- la fase `apnea` incluye `apnea/no_apnea`, `AHI`, severidad por paciente y biomarcadores adicionales solo si los datos los soportan.
- el plan queda alineado con los requerimientos de evaluacion interna, externa, reproducibilidad, metricas por paciente y comparacion con literatura.

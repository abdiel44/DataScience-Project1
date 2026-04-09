# Proyecto -1: Machine Learning para clasificación de etapas de sueño y detección de apnea

**Universidad Interamericana PR – Bayamón | Ciencia de Datos**

## Resume del proyecto
• Uso exclusivo de EEG de un solo canal (frontal o central estándar).
• Aplicación y comparación de tres familias de modelos tradicionales: SVM-RBF, Random Forest/baging y Boosting tipo AdaBoost/XGBoost/Gradient Boosting.
• Validación subject-wise y experimentos cross-dataset para medir robustez y generalización.        
• Comparación crítica con baselines publicados y propuesta de una mejora metodológica.

## Contexto del proyecto
La Apnea Obstructiva del Sueño (AOS) produce interrupciones repetidas de la respiración durante el sueño y su diagnóstico clínico se apoya en estudios Polisomnografía (PSG). En años recientes ha crecido el interés por sistemas automáticos más simples que empleen señales de Electroencefalografía (EEG) monocanal (1 solo canal) para clasificar etapas de sueño y detectar apnea, con el fin de reducir complejidad instrumental y facilitar escenarios de monitoreo simplificado.

## Objetivo general
Aplicar tres algoritmos tradicionales de machine learning: SVM con kernel (busque el mejor), Random Forest y variantes tipo boosting — para dos tareas: (1) clasificación de etapas de sueño y (2) detección de apnea usando únicamente EEG de un solo canal, con comparación frente a trabajos reconocidos (baselines) publicados sobre los mismos conjuntos de datos.

![Flujo experimental sugerido para el proyecto](https://i.imgur.com/3Q5z5QG.png)

*Figura 1. Flujo experimental sugerido para el proyecto*

## Objetivos específicos
• Construir pipelines reproducibles de preprocesamiento, extracción de características, entrenamiento y evaluación.
• Implementar al menos tres familias de modelos: SVM-RBF, Random Forest, y una variante boosting (o ensamblada)
• Evaluar desempeño dentro de cada dataset mediante validación subject-wise k-fold.
• Evaluar generalización cross-dataset entrenando en una o varias bases y probando en otra.
• Comparar resultados propios con 6–10 referencias científicas relevantes y con al menos 3 baselines bien documentados.
• Analizar críticamente las limitaciones observadas y proponer un nuevo método que pudiera superar el baseline obtenido.

## Modelos a utilizar

| Modelo | Sugerencias mínimas | Observación |
|---|---|---|
| SVM (RBF) | Buscar C, gamma, class_weight y estrategia multiclas | Especialmente útil con features bien normalizados y seleccionados |
| Random Forest / bagging | n_estimators, max_depth, max_features, min_samples_leaf | Debe incluir importancia de features |
| Boosting tipo AdaBoost/XGBoost/Gradient Boosting | Comparar contra RF puro con mismo protocolo | Muy recomendado para clases desbalanceadas |
| HMM | Probar HMM directo sobre features y/o HMM como suavizado de secuencias | La matriz de transición debe analizarse e interpretarse |

## Conjuntos de datos
El proyecto utilizará registros públicos de sueño restringidos a un único canal EEG (frontal o central estándar), siguiendo protocolos por sujeto para evitar fuga de información.

| Dataset | Uso esperado | Etiquetas | Tener en cuenta: |
|---|---|---|---|
| Sleep-EDF Expanded | Base principal para desarrollo y baseline | W / N1 / N2 / N3 / REM; apnea si la anotación está disponible en la partición usada | Documentar canal elegido, mapeo de etiquetas y limpieza |
| MIT-BIH PSG | Base complementaria para apnea/sueño | Etiqueta Apnea/no-apnea por ventana; estadios si existen en la partición seleccionada | Explicar claramente cómo se sincronizan anotaciones con ventanas |
| SHHS | Base amplia para generalización | Apnea/no-apnea, severidad global y staging según disponibilidad | Ideal para estudiar cambio de dominio y cohortes heterogéneas |
| St. Vincent’s University Hospital | Base clínica adicional | Apnea/no-apnea y/o staging según el subconjunto trabajado | Especificar partición, acceso y canal equivalente al estándar |
| ISRUUC-Sleep | Base complementaria para clasificación etapas(staging) | W / N1 / N2 / N3 / REM | Útil para contrastar con Sleep-EDF y estudiar generalización |

*Los datasets son públicos y los puede bajar de Internet crudos (raw) o ya listos para trabajar. (HuggingFace o Kaggle recomendados).*

## Diseño experimental

1. Protocolos subject-wise: cada sujeto debe aparecer exclusivamente en entrenamiento/validación o en prueba.
2. Validación k-fold dentro de cada dataset, reportando media y desviación estándar.
3. Experimentos cross-dataset: entrenar en uno o varios datasets y probar en otro, sin ajustar hiperparámetros sobre el conjunto de prueba destino.
4. Todas las etapas del pipeline que aprendan parámetros (normalización, selección de features, balanceo, calibración) deben ajustarse únicamente con el fold de entrenamiento.
5. Se debe fijar semilla aleatoria y registrar configuración para reproducibilidad.

![Diagramas de protocolos](https://i.imgur.com/3Q5z5QG.png)

*Figura 2. Los dos protocolos mínimos exigidos por el proyecto.*

## Preprocesamiento y extracción de características
Se espera que al menos consideren:
• Segmentación en ventanas consistentes con el etiquetado.
• Filtrado y/o remuestreo apropiado para el canal EEG seleccionado.
• Normalización por sujeto o por registro, documentando el procedimiento.
• Técnica para extracción de features (características)
• Estrategias para desbalance de clases: class weighting, undersampling, oversampling, SMOTE u otras.
• Análisis de artefactos/ruido: cómo se detectan, excluyen o modelan.

## Métricas y análisis estadístico
- **Para apnea/no-apnea:** Accuracy, sensibilidad, especificidad y AUC-ROC. Exactitud en la clasificación de severidad por paciente.
- **Para estadios de sueño:** Accuracy global, macro-F1 y kappa de Cohen. Análisis por clase con énfasis especial en N1.
- **Métricas por paciente:** Sensibilidad y especificidad para puntos de corte de AHI que el equipo adopte y documente.

## Requisitos de implementación y reproducibilidad
• Se debe implementar el código pensando en que va a hacer reproducible. **OBLIGATORIO: entorno de Desarrollo usando Docker para entornos reproducibles.**
• Todo resultado debe guardarse en formato CSV; toda figura final debe guardarse en PNG.
• El proyecto debe registrar hiperparámetros, seeds, tiempos de entrenamiento y particiones usadas. 
• Se debe incluir un archivo *.txt con scripts ejemplos de ejecución.
• El README.md debe explicar dependencias, estructura de carpetas, datasets y pasos para reproducir.

### Anexo: Etapas del Sueño (AASM)
| N | Etapa | Descripción | Código |
|---|---|---|---|
| 1 | W | Despierto, Predominan (más del 50%) las ondas Alfa | 0 |
| 2 | N1 (1-5%) | Adormecimiento. Ondas Alfa y Theta de baja amplitud | 1 |
| 3 | N2 (45-55%) | Sueño ligero. Complejos K y husos del sueño | 2 |
| 4 | N3 (15-25%) | Sueño profundo. Ondas Delta | 3 |
| 5 | REM (20-25%)| Movimientos oculares rápidos, ocurre el sueño vívido | 4 |

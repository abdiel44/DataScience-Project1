# Flujo completo (for dummies): preprocesamiento → entrenamiento

Guía **paso a paso**, alineada con **cómo está montado este repo**: primero obtienes una **tabla** (CSV), luego el `**main.py` la “preprocesa”** (limpieza, codificación, balanceo, escalado, etc.) y **por último entrenas** leyendo ese CSV **ya preprocesado**.

---

## 1. Idea general (léelo una vez)

Imagina **tres cajas**:

1. **Datos crudos**
  Archivos grandes en `data\raw\` (PhysioNet, carpetas WFDB, EDF, etc.). A veces **no** son un CSV listo para machine learning.
2. **Tabla “en bruto” pero ya filas/columnas**
  Un **CSV** donde cada fila es una ventana (p. ej. 30 s) o un evento, con columnas numéricas (features) y columnas de sujeto / etiqueta.  
   En este proyecto eso sale del **export WFDB** (`--export-epochs`) o de **ingesta** (`--source ...`) o de un CSV que ya tengas.
3. **Tabla preprocesada**
  Otro **CSV** que sale de `**python src\main.py ...`** con `--encoding-spec` (y opciones de escalado, etc.). Es la que **deberías usar para entrenar** si quieres cumplir el flujo completo “preprocesamiento → entrenamiento”.

**Entrenamiento (Fase E)** = otro comando que **solo lee un CSV** (el que tú indiques en el YAML). **No** vuelve a pasar por `main.py`. Por eso es **importante** que en el YAML pongas la ruta del CSV **preprocesado** (p. ej. `..._prep.csv`), no mezclar sin querer el export crudo.

---

## 2. Qué haces **una sola vez** (tu máquina)

### 2.1 Instalar Python

Necesitas **Python 3.11** (o el que indique el curso). Si no tienes ninguno, descárgalo del sitio oficial e instálalo marcando “Add Python to PATH” si te lo ofrece.

### 2.2 Abrir la carpeta del proyecto

En el Explorador de archivos entra a tu repo, por ejemplo:

`C:\Users\Steven\University\2026\DataScience\FinalProjectTemplate`

Ahí debes ver carpetas como `src`, `config`, `data`, `scripts`.

### 2.3 Crear el entorno virtual e instalar librerías

Abre **PowerShell**, ve a la raíz del proyecto y ejecuta **exactamente**:

```powershell
cd C:\Users\Steven\University\2026\DataScience\FinalProjectTemplate
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- `**cd**`: “estoy en la carpeta del proyecto”.
- `**python -m venv .venv**`: crea una carpeta `.venv` con un Python aislado.
- `**Activate.ps1**`: activa ese entorno (en la terminal verás algo como `(.venv)` al inicio de la línea).
- `**pip install ...**`: instala scikit-learn, pandas, wfdb, xgboost, etc.

**Tienes que hacer esto una vez por proyecto** (o si borras `.venv`). No hace falta repetirlo “por cada dataset”.

Si `Activate.ps1` da error de política de ejecución en PowerShell, es un tema de Windows; el profesor o la documentación del curso suele indicar `Set-ExecutionPolicy` temporal o usar **CMD** con `.\.venv\Scripts\activate.bat`.

---

## 3. ¿Hay que repetir todo “por cada dataset”?

**Sí, en la práctica**, porque cada corpus (MIT-BIH, SHHS, ISRUC, …) tiene:

- carpetas distintas en `data\raw\`,
- un **JSON distinto** en `config\datasets\*.json` para el preprocesamiento,
- a veces un **CSV de export** distinto,
- y un **YAML de experimento** (o rutas distintas dentro del mismo patrón).

**No** hace falta crear otro `venv` por dataset. Solo repites: *sacar CSV tabular → preprocesar → ajustar YAML → entrenar*.

---

## 4. Flujo completo en 4 fases (por cada dataset que uses)

Usamos nombres de ejemplo para MIT-BIH estadificación; tú puedes cambiar rutas y nombres, pero **la secuencia** es la misma.

### Fase A — Tener datos crudos

**Qué haces tú (manual):**

1. Crea `data\raw\` si no existe.
2. Descarga el dataset (PhysioNet, etc.) y descomprímelo **dentro** de `data\raw\` como indique la documentación del dataset (estructura de carpetas que espera el código).

**Comando:** ninguno todavía, solo organizar archivos.

---

### Fase B — De crudo a **CSV tabular** (aún no “preprocesado” del pipeline)

Aquí el objetivo es obtener algo como `mitbih_sleep_stages.csv`: filas = épocas, columnas = features + `record_id` + `sleep_stage`, etc.

**Opción 1 — Export WFDB (MIT-BIH o SHHS)**  
**Comando ejemplo MIT-BIH:**

```powershell
cd C:\Users\Steven\University\2026\DataScience\FinalProjectTemplate
.\.venv\Scripts\Activate.ps1

python src\main.py --export-epochs mit-bih-psg --raw-root data\raw `
  --output-stages data\processed\mitbih_sleep_stages.csv `
  --output-events data\processed\mitbih_respiratory_events.csv
```

- Eso **no** pasa por limpieza/codificación completa del pipeline largo: **solo escribe dos CSV** de épocas y termina.
- Para **SHHS** cambias `mit-bih-psg` por `shhs-psg` y los nombres de salida.

**Opción 2 — Otra fuente (ingesta con `--source`)**  
Si tu proyecto usa ISRUC u otro origen, el patrón es otro comando con `--source` y `--output`; el detalle está en `reproducir.txt` y en el README. La idea es la misma: al final tienes **un** CSV en `data\processed\`.

**¿Por dataset?** Sí: MIT-BIH y SHHS son **dos comandos distintos** si quieres **dos** CSV distintos.

---

### Fase C — **Preprocesamiento** (lo que tú llamas “datos preprocesados”)

Aquí entra el **pipeline** de `main.py` en serio:

**Orden interno (automático):** limpieza → codificación → (opcional) balanceo → (opcional) escalado → (opcional) reducción de dimensionalidad → **un CSV de salida**.

**Qué necesitas:**

- `--input`: el CSV **tabular** de la fase B (export o ingesta).
- `--output`: **otro archivo**, convención `*_prep.csv`, para no pisar el anterior.
- `--encoding-spec`: un **JSON** de `config\datasets\` que corresponda a **ese** dataset y **esa** tarea (ver tabla en `docs\CONFIG_DATASETS.md`).
- `--target-col`: nombre de la columna objetivo (ej. `sleep_stage`).
- `--task`: un **nombre corto** solo para reportes/carpetas (puedes poner algo como `mitbih_staging`).

**Comando tipo (MIT-BIH estadificación):**

```powershell
cd C:\Users\Steven\University\2026\DataScience\FinalProjectTemplate
.\.venv\Scripts\Activate.ps1

python src\main.py `
  --input data\processed\mitbih_sleep_stages.csv `
  --output data\processed\mitbih_sleep_stages_prep.csv `
  --encoding-spec config\datasets\mitbih_sleep_stages.json `
  --target-col sleep_stage `
  --task mitbih_staging `
  --write-cleaning-report
```

**Para eventos respiratorios MIT-BIH** usarías el otro CSV y el otro JSON, por ejemplo `--input ...mitbih_respiratory_events.csv`, `--encoding-spec config\datasets\mitbih_respiratory_events.json`, `--target-col event_tokens`, y otro `--output` y `--task`.

**¿Por dataset y por tarea?**  

- **Sí**, un comando (o varios si haces staging + events) **por cada combinación** “tabla de entrada + JSON + objetivo” que quieras usar en el informe.

**Opcional pero útil:**

- `--run-eda`: EDA sobre datos limpios.
- `--run-eda --run-eda-processed`: EDA sobre la tabla **final** ya codificada/escalada (Fase D).
- `--balance-method`, `--scale-method`, etc.: según diseño experimental (documentado en `DISENO_PREPROCESAMIENTO.md`).

Al terminar, el archivo que importa para entrenar es, en este ejemplo:

`data\processed\mitbih_sleep_stages_prep.csv`

---

### Fase D — **Entrenamiento** (Fase E)

El entrenamiento **no ejecuta** el preprocesamiento. Lee un YAML y un CSV.

**Qué haces tú (manual):**

1. Abre un preset en `config\experiments\` o copia `config\experiment_train.example.yaml` a `config\mi_exp.yaml`.
2. Ajusta **al menos**:
  - `train_csv`: debe apuntar al CSV **preprocesado** de la fase C, p. ej.  
   `data/processed/mitbih_sleep_stages_prep.csv`
  - `subject_column` y `target_column`: deben coincidir con las columnas **después** del preprocesamiento (el target codificado puede seguir llamándose igual; revisa el CSV con Excel o un script corto si dudas).
  - `task`: `multiclass` para etapas de sueño / varias clases; `binary` solo si tu columna objetivo es apnea en **0/1** (ver validaciones del runner).
  - Si el CSV tiene **muchas señales** (`*_mean`, `*_std`), puede que debas rellenar `feature_include` con **un solo canal** EEG (requisito del proyecto).

**Comando:**

```powershell
cd C:\Users\Steven\University\2026\DataScience\FinalProjectTemplate
.\.venv\Scripts\Activate.ps1

python scripts\run_phase_e_cv.py --config config\mi_exp.yaml
```

(Si usas un preset, cambia `config\mi_exp.yaml` por `config\experiments\mitbih_sleep_stages.yaml` **pero** entonces edita ese preset para que `train_csv` sea el `*_prep.csv`, no el export sin preprocesar, si quieres el flujo “correcto” preprocesado → train.)

**Salida:** carpetas bajo `reports\experiments\<nombre_experimento>\` con métricas, predicciones, figuras, etc.

**¿Por dataset?**  

- Si entrenas **solo MIT-BIH**, un YAML (y un `train_csv` prep).  
- Si haces **cross-dataset** (entrenar en A, evaluar en B), necesitas **dos** CSV preprocesados y un YAML con `cross_dataset: true` y `eval_csv` al prep de B.

---

## 5. Errores típicos “no entiendo qué pasó”


| Síntoma                                  | Causa frecuente                                                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `train_csv not found`                    | Ruta mal escrita o no generaste el CSV; el runner busca también junto al YAML.                               |
| Error de varios canales `*_mean`/`*_std` | Falta `feature_include` con un solo stem de señal, o `allow_multi_channel_features: true` si es a propósito. |
| Error con binario y 0/1                  | `task: binary` con etiquetas que no son 0/1; crea columna 0/1 o usa `binary_require_zero_one_labels: false`. |
| “Entrené pero usé el CSV equivocado”     | Apuntaste al export sin `--encoding-spec` en lugar de `*_prep.csv`.                                          |


---

## 6. Mini-resumen en una frase

**Una vez** instalas el entorno; **por cada dataset/tarea** haces: **crudo → CSV tabular (export/ingesta) → CSV prep (`main.py` + JSON) → entrenamiento (`run_phase_e_cv.py` + YAML con `train_csv` = prep)**.

---

## 7. Checklist imprimible

1. [ ] `.\.venv\Scripts\Activate.ps1` y `pip install -r requirements.txt`
2. [ ] Datos en `data\raw\`
3. [ ] CSV tabular creado (`--export-epochs` o `--source` u otro)
4. [ ] CSV prep creado (`main.py` con `--encoding-spec` correcto)
5. [ ] YAML con `train_csv` = CSV prep y columnas revisadas
6. [ ] `python scripts\run_phase_e_cv.py --config ...`


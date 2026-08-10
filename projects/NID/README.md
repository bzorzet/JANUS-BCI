# Proyecto NID

Paper: *[título del paper]*
Estado: en desarrollo

## Descripción

Clasificación de Motor Imagery (MI-BCI) comparando modelos DL y ML
clásicos sobre múltiples datasets públicos con distintas estrategias
de preprocesamiento (CAR, ICA) y validación (Within-Subject,
Leave-One-Subject-Out).

## Datasets

| Dataset | Sesiones | Canales | sfreq |
|---|---|---|---|
| Cho2017 | session_1 | 27 (bilateral) | 128 Hz |
| Lee2019 | session_1, session_2 | 21 (bilateral) | 125 Hz |
| Dreyer2023A | session_1 | 21 (bilateral) | 128 Hz |

## Estrategias de validación

| `strategy_name` | Descripción |
|---|---|
| `WS-Standard` | Within-subject, split fijo 60/20/20 |
| `WS-LOSO` | Leave-One-Subject-Out |

## Preprocesamiento

| Nombre | Archivo config | Descripción |
|---|---|---|
| `CAR-preproc` | `configs/preprocessing/2026-NID_*_CAR-preproc.json` | Referencia CAR, filtro 0.5-40 Hz |
| `ICA10-preproc` | `configs/preprocessing/2026-NID_*_ICA10-preproc.json` | 10 componentes ICA + CAR |
| `original-preproc` | `configs/preprocessing/2026-NID_*_original-preproc.json` | Referencia Fz |

## Modelos

| Modelo | Tipo | `recipe_name` |
|---|---|---|
| CTNet | DL | `CTNet` |
| ATCNet | DL | `ATCNet` |
| EEGNetv4 | DL | `EEGNetv4` |
| SincNet | DL | `SincNet` |
| FBCSP | ML | `FBCSP` |
| CSP-alt | ML | `CSP-alt` |

## Estructura de carpetas

```
NID/
├── configs/
│   ├── training/
│   │   ├── WS-Standard/     # configs de entrenamiento WS estándar
│   │   └── WS-LOSO/         # configs de entrenamiento LOSO
│   └── preprocessing/       # configs de preprocesamiento por dataset
├── generators/              # scripts privados de generación de configs
│   ├── config_generator.py  # genera todos los JSONs automáticamente
│   ├── models.py            # registry de arquitecturas y params
│   ├── datasets.py          # registry de datasets, canales y sfreq
│   └── preprocessings.py   # registry de pipelines de preprocesamiento
├── analysis/                # scripts de análisis de resultados
│   ├── 01_tagging_subjects.py
│   ├── 02_summary_metrics.py
│   └── plot_comparisons.py
└── commands/                # .sh para lanzar experimentos en batch
    ├── WS-Standard_experiments.sh
    └── WS-LOSO_experiments.sh
```

## Convención de nombrado de configs

Ver `PROTOCOL.md` sección 5 del repo raíz para la convención completa.

**Training:**
```
{project}_{strategy}_{model}_{dataset}_{label}.json
2026-NID_WS-Standard_CTNet_Cho2017_CAR-Bilateral-Full.json
```

**Preprocessing:**
```
{project}_{dataset}_{session}_{preprocessing-name}.json
2026-NID_Cho2017_s1_CAR-preproc.json
```

## Cómo correr experimentos

```bash
# 1. Preprocesar un dataset (si no está preprocesado)
python scripts/run_production.py \
  --config projects/NID/configs/preprocessing/2026-NID_Cho2017_s1_CAR-preproc.json

# 2. Entrenar un modelo
python scripts/run_production.py \
  --config projects/NID/configs/training/WS-Standard/2026-NID_WS-Standard_CTNet_Cho2017_CAR-Bilateral-Full.json

# 3. O lanzar todos los experimentos de una estrategia en batch
bash projects/NID/commands/WS-Standard_experiments.sh
```

## Nota sobre el repo público

La carpeta `generators/` es privada y no se incluye en el
repositorio del paper. Solo se publican los JSONs generados
en `configs/` y los scripts de `analysis/`.

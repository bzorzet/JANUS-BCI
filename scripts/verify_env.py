"""
Verifica que el entorno dentro del contenedor tiene todo lo que
JANUS-BCI necesita. Correrlo así:

    docker run --rm --gpus all janus-bci python scripts/verify_env.py

Sin GPU (solo CPU):
    docker run --rm janus-bci python scripts/verify_env.py
"""
import sys

VERDE = "\033[92m"
ROJO = "\033[91m"
AMARILLO = "\033[93m"
RESET = "\033[0m"

ok = []
warn = []
fail = []


def check(nombre, fn):
    try:
        resultado = fn()
        ok.append(f"{VERDE}✓{RESET} {nombre}: {resultado}")
    except Exception as e:
        fail.append(f"{ROJO}✗{RESET} {nombre}: {e}")


def check_warn(nombre, fn):
    try:
        resultado = fn()
        ok.append(f"{VERDE}✓{RESET} {nombre}: {resultado}")
    except Exception as e:
        warn.append(f"{AMARILLO}⚠{RESET} {nombre}: {e}")


# Python
check("Python", lambda: sys.version.split()[0])

# PyTorch
def check_torch():
    import torch
    return f"torch {torch.__version__}"
check("PyTorch", check_torch)

# CUDA
def check_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible (normal si no hay GPU en esta máquina)")
    return f"CUDA {torch.version.cuda} · {torch.cuda.get_device_name(0)}"
check_warn("CUDA / GPU", check_cuda)

# MNE
def check_mne():
    import mne
    return f"mne {mne.__version__}"
check("MNE", check_mne)

# mne-icalabel
def check_icalabel():
    import mne_icalabel
    return f"mne-icalabel {mne_icalabel.__version__}"
check("mne-icalabel", check_icalabel)

# Pyriemann
def check_pyriemann():
    import pyriemann
    return f"pyriemann {pyriemann.__version__}"
check("pyriemann", check_pyriemann)

# Braindecode
def check_braindecode():
    import braindecode
    return f"braindecode {braindecode.__version__}"
check("braindecode", check_braindecode)

# Scikit-learn
def check_sklearn():
    import sklearn
    return f"scikit-learn {sklearn.__version__}"
check("scikit-learn", check_sklearn)

# Scipy
def check_scipy():
    import scipy
    return f"scipy {scipy.__version__}"
check("scipy", check_scipy)

# Pandas
def check_pandas():
    import pandas
    return f"pandas {pandas.__version__}"
check("pandas", check_pandas)

# MLFlow
def check_mlflow():
    import mlflow
    return f"mlflow {mlflow.__version__}"
check("MLFlow", check_mlflow)

# SQLAlchemy
def check_sqlalchemy():
    import sqlalchemy
    return f"sqlalchemy {sqlalchemy.__version__}"
check("SQLAlchemy", check_sqlalchemy)

# pydantic-settings (para paths.py)
def check_pydantic():
    import pydantic_settings
    return f"pydantic-settings {pydantic_settings.VERSION}"
check("pydantic-settings", check_pydantic)

# Impresión de resultados
print("\n─── Verificación del entorno JANUS-BCI ───\n")
for line in ok:
    print(line)
for line in warn:
    print(line)
for line in fail:
    print(line)

print()
if fail:
    print(f"{ROJO}Hay {len(fail)} error(es) crítico(s). Revisar el Dockerfile / environment.yml.{RESET}")
    sys.exit(1)
elif warn:
    print(f"{AMARILLO}Todo crítico OK. {len(warn)} advertencia(s) no crítica(s) (ej. sin GPU en esta máquina).{RESET}")
else:
    print(f"{VERDE}Entorno completo y listo.{RESET}")

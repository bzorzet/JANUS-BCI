#!/usr/bin/env bash
set -euo pipefail

# =============================================================
# Setup de JANUS-BCI-env (local, sin Docker)
# Replica environment.gpu.yml + las 2 capas del Dockerfile
# =============================================================

ENV_NAME="JANUS-BCI-env"

# ── Capa 1: stack conda (sin PyTorch), igual a environment.gpu.yml ──
conda create -n "${ENV_NAME}" -y \
  -c pytorch -c nvidia -c conda-forge -c defaults \
  python=3.11.10 \
  scipy=1.17.0 \
  mne=1.11.0 \
  pandas=2.3.3 \
  scikit-learn=1.6.1 \
  numpy \
  matplotlib \
  seaborn \
  libstdcxx-ng=12 \
  jupyterlab \
  ipykernel \
  ipywidgets \
  python-dotenv=1.0.1 \
  pip

# Activar para las capas siguientes
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ── Capa 2: PyTorch con versión cu118 exacta, vía índice oficial ──
# (igual que en el Dockerfile: separado del yml para usar --index-url)
pip install --no-cache-dir \
  torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
  --index-url https://download.pytorch.org/whl/cu118

# ── Capa 3: bloque pip del environment.gpu.yml ──
pip install --no-cache-dir \
  setuptools==75.6.0 \
  mlflow==2.13.0 \
  sqlalchemy==2.0.30 \
  pydantic==2.7.4 \
  pydantic-settings==2.3.4 \
  mne-icalabel==0.8.1 \
  pyriemann==0.7 \
  rich==13.9.4

# ── Capa 4 (extra, no está en el yml pero sí la usaba tu entorno viejo) ──
# PyG + herramientas de entrenamiento, si el proyecto nuevo las sigue usando.
# Comentado por default: descomentar si JANUS-BCI también necesita PyG.
# pip install --no-cache-dir \
#   pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
#   -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
# pip install --no-cache-dir einops==0.8.0 torch_geometric==2.5.3 torchcam==0.4.0

echo ""
echo "=== Setup completo. Verificando... ==="
python -c "
import torch, scipy, mne, sklearn, pyriemann
print(f'GPU disponible: {torch.cuda.is_available()}')
print(f'Torch:      {torch.__version__}')
print(f'SciPy:      {scipy.__version__}')
print(f'MNE:        {mne.__version__}')
print(f'Sklearn:    {sklearn.__version__}')
print(f'PyRiemann:  {pyriemann.__version__}')
"

echo ""
echo "=== Diagnóstico preventivo CXXABI (ver RESUMEN_issue_entorno.md) ==="
ENV_PATH="$(conda info --base)/envs/${ENV_NAME}"
if strings "${ENV_PATH}/lib/libstdc++.so.6" | grep -q "CXXABI_1.3.15"; then
  echo "OK: CXXABI_1.3.15 presente en libstdc++ del entorno (${ENV_PATH}/lib)."
  echo "Si igual aparece el ImportError al correr scripts, es problema de"
  echo "orden de búsqueda de librerías dinámicas. Solución:"
  echo "  export LD_LIBRARY_PATH=${ENV_PATH}/lib:\$LD_LIBRARY_PATH"
else
  echo "ALERTA: CXXABI_1.3.15 NO encontrado en libstdc++ del entorno."
  echo "Revisar/reinstalar libstdcxx-ng dentro de ${ENV_NAME}."
fi
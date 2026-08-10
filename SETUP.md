# Setup — JANUS-BCI

Guía de instalación desde cero. Seguila en orden la primera vez.
Una vez completada, el flujo diario está en `CLAUDE.md`.

---

## Requisitos previos

- Ubuntu 22.04 (o derivado, ej. Zorin OS)
- Git instalado (`sudo apt install git`)
- GPU NVIDIA (opcional — sin GPU igual funciona con el perfil `cpu`)

---

## 1. Instalar Docker

```bash
# Agregar el repositorio oficial de Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
https://download.docker.com/linux/ubuntu jammy stable" | \
sudo tee /etc/apt/sources.list.d/docker.list

sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Agregar tu usuario al grupo docker (evita usar sudo cada vez)
sudo usermod -aG docker $USER

# Aplicar el grupo sin cerrar sesión (solo para esta terminal)
newgrp docker

# Verificar
docker --version
```

---

## 2. Instalar NVIDIA Container Toolkit (solo si tenés GPU)

Si no tenés GPU NVIDIA, saltá al paso 3.

```bash
# Agregar el repositorio de NVIDIA
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit

# Configurar Docker para que use el runtime de NVIDIA
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verificar que Docker ve la GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
# Tiene que mostrar tu GPU con nvidia-smi
```

---

## 3. Clonar el repositorio

```bash
git clone <URL-del-repo> JANUS-BCI
cd JANUS-BCI
```

---

## 4. Configurar las rutas locales

Copiá el archivo de ejemplo y completá las rutas de tu máquina.
Este archivo **nunca se versiona** — cada máquina tiene el suyo.

```bash
cp .env.example .env
nano .env   # o el editor que prefieras
```

Las rutas que tenés que completar:

```bash
JANUS_REPO_ROOT=/home/<tu_usuario>/JANUS-BCI   # donde clonaste el repo (SSD)
JANUS_DATA_ROOT=/mnt/hdd/janus_data            # datos crudos/procesados (HDD)
JANUS_RESULTS_ROOT=/mnt/hdd/janus_results      # resultados de experimentos
JANUS_SANDBOX_ROOT=/mnt/hdd/janus_sandbox      # pruebas rápidas descartables
```

Verificá que las carpetas existen (créalas si no):

```bash
mkdir -p $JANUS_DATA_ROOT $JANUS_RESULTS_ROOT $JANUS_SANDBOX_ROOT
```

---

## 5. Construir la imagen Docker

Elegí según tu hardware:

```bash
# Con GPU (laptop o servidor con NVIDIA)
docker compose --profile gpu build bci-gpu

# Sin GPU (solo CPU)
docker compose --profile cpu build bci-cpu
```

La primera vez tarda ~20-30 minutos — descarga Ubuntu + CUDA +
Miniforge + todas las librerías de BCI. Las siguientes veces usa
caché y es mucho más rápido.

---

## 6. Verificar el entorno

```bash
# Con GPU
docker compose --profile gpu run --rm bci-gpu python scripts/verify_env.py

# Sin GPU
docker compose --profile cpu run --rm bci-cpu python scripts/verify_env.py
```

Todos los ítems tienen que mostrar ✓. El ⚠ de CUDA solo aparece
si corrés el perfil cpu — es esperado.

---

## 7. Levantar el dashboard de MLFlow

```bash
docker compose up mlflow
```

Abrí `http://localhost:5000` en el navegador. Acá vas a ver todos
los experimentos trackeados de forma visual.

---

## 8. VS Code Dev Containers (recomendado para desarrollo diario)

### Requisitos
- VS Code instalado
- Extensión **Dev Containers** (`ms-vscode-remote.remote-containers`)
- La imagen ya construida (paso 5)

### Primer uso
1. Abrí VS Code en la carpeta del repo: `code ~/JANUS-BCI`
2. VS Code detecta `.devcontainer/devcontainer.json` y muestra una
   notificación — click en **"Reopen in Container"**. Si no aparece,
   `Ctrl+Shift+P` → **"Dev Containers: Reopen in Container"**.
3. VS Code arranca el servicio `bci-gpu` de `docker-compose.yml`,
   instala las extensiones automáticamente, y corre
   `scripts/verify_env.py` — revisá el panel de log del contenedor
   para confirmar que todo salió ✓.
4. El botón Run y el debugger ya funcionan apuntando al Python de
   `/opt/conda/envs/janus-bci/bin/python`.

### Jupyter Notebooks dentro del contenedor
Abrí cualquier `.ipynb`, seleccioná el kernel **"janus-bci"**, corré
las celdas normalmente — todo corre dentro del contenedor con GPU.

### Cambiar de GPU a CPU
Editá `.devcontainer/devcontainer.json`:
```json
"service": "bci-cpu"
```
La GPU se configura enteramente en `docker-compose.yml` (bloque
`deploy.resources` del servicio `bci-gpu`) — el `devcontainer.json`
solo elige a qué servicio conectarse, no necesita ningún flag de GPU
propio (`runArgs` se ignora cuando se usa `dockerComposeFile`, ver
`BUGS.md`). Después de cambiar el servicio, reconstruí:
`Ctrl+Shift+P` → **"Dev Containers: Rebuild Container"**.

---

## Uso diario

Ver `CLAUDE.md` para los comandos de uso diario y el protocolo
de trabajo.

---

## Solución de problemas comunes

**`permission denied` al correr docker**
```bash
sudo usermod -aG docker $USER && newgrp docker
```

**`Unable to locate package nvidia-container-toolkit`**
El repositorio de NVIDIA no está agregado. Seguí el paso 2 desde
el principio — el paquete no viene en los repos de Ubuntu por defecto.

**`$(ARCH)` literal en nvidia-container-toolkit.list**
El script de NVIDIA no expandió la variable. Corregilo manualmente:
```bash
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
```

**`curl: (22) The requested URL returned error: 404` en docker build**
Mambaforge fue discontinuado. El Dockerfile actual ya usa
Miniforge3 — si ves este error, verificá que tu Dockerfile tenga
`Miniforge3-Linux-x86_64.sh` y no `Mambaforge-Linux-x86_64.sh`.

**CUDA no disponible dentro del contenedor**
Verificá que usaste `--profile gpu` y que el NVIDIA Container
Toolkit está instalado y configurado:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## Ítems abiertos (no soportados todavía)

- **Windows** — Docker Desktop + WSL2 + drivers NVIDIA para WSL
  tienen un flujo de instalación distinto al de Linux. No está
  documentado ni probado todavía.
- **Mac** — Docker Desktop para Mac no tiene acceso a GPU NVIDIA
  (Apple usa Metal/MPS). El perfil `cpu` probablemente funcione
  pero no está probado. Para entrenar modelos serios se recomienda
  usar un servidor Linux con GPU.

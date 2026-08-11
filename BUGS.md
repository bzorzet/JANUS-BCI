# Bugs — JANUS-BCI

Bitácora cronológica de problemas reales encontrados y cómo se
resolvieron. A diferencia de la sección "Solución de problemas
comunes" de `SETUP.md` (que es la versión limpia y prescriptiva
para alguien que arranca de cero), este documento es el registro
crudo de qué pasó, por qué, y qué se probó.

Sirve para dos cosas: no repetir el mismo debugging dos veces, y
tener contexto histórico de por qué un archivo quedó como quedó.

## Cómo agregar una entrada

Copiar el template, completar, agregar arriba de todo (orden
cronológico inverso — lo más reciente primero).

```markdown
## [YYYY-MM-DD] Título corto

**Contexto:** dónde/haciendo qué pasó
**Síntoma:** el error exacto
**Causa:** por qué pasó
**Solución:** qué lo resolvió
**Tags:** #docker #conda #gpu ...
```

---

## [2026-08-11] Dev Containers — problemas de compatibilidad de versiones entre máquinas

**Contexto:** Intentando usar VS Code Dev Containers desde tres
contextos distintos: SSH desde laptop, físicamente en la PC del
instituto, y vía AnyDesk.

**Síntoma:**
```
Error al configurar el contenedor.
Exit code 1
```
El contenedor quedaba en estado `Created` sin llegar a `Running`.

**Causa:** VS Code Dev Containers instala un "VS Code Server" dentro
del contenedor que tiene que coincidir exactamente con la versión
del cliente VS Code. Con versiones distintas entre la sesión SSH
y la sesión física, o tras una actualización de VS Code, el servidor
no matchea y falla silenciosamente con `Exit code 1`.

**Solución adoptada:** Abandonar Dev Containers para desarrollo
diario. Usar el flujo más simple:
- Desarrollo: VS Code Remote SSH + intérprete conda del host
- Producción: `docker compose --profile gpu run bci-gpu`

Docker sigue siendo parte del proyecto pero para su propósito
original — reproducibilidad y ejecución en servidores — no como
entorno de desarrollo diario.

**Lección:** Dev Containers agrega una capa de complejidad que solo
vale la pena cuando el equipo es grande y la consistencia del entorno
entre muchos desarrolladores es crítica. Para investigación en
solitario o equipo pequeño, conda local + Docker para producción
es más robusto y con menos fricción.

**Tags:** #docker #devcontainers #vscode


---
## 2026-08-10 — `EROFS` al preprocesar dentro del contenedor: env var confunde path del host con path del contenedor

**Síntoma:** OSError: [Errno 30] Read-only file system: '/data/JANUS-BCI-results' al importar `src.preprocessing` (dispara `PATHS = JanusPaths()` en `src/utils/paths.py`),
corriendo `scripts/run_production.py` dentro de `bci-gpu`/`bci-cpu`.

**Causa raíz:** `docker-compose.yml` monta `${JANUS_RESULTS_ROOT}:/results`, pero
`env_file: .env` además propaga `JANUS_RESULTS_ROOT` —con el valor del host,
`/data/JANUS-BCI-results`— como variable de entorno *adentro* del contenedor.
`PATHS` lee esa variable para saber dónde escribir, así que terminaba armando
rutas contra `/data/JANUS-BCI-results`, que ahí adentro no es ningún mount —
es una subcarpeta de `/data`, el bind **read-only** de `JANUS_DATA_ROOT`.

**Fix:** agregar un bloque `environment:` a `bci-gpu`/`bci-cpu` que pisa las
variables de `env_file` con los paths tal como se ven adentro del contenedor
(Compose aplica `environment:` después de `env_file:`, gana el valor fijo):
```yaml
environment:
  JANUS_REPO_ROOT: /workspace
  JANUS_DATA_ROOT: /data
  JANUS_RESULTS_ROOT: /results
  JANUS_SANDBOX_ROOT: /sandbox
```

**Cómo se detectó:** `docker inspect <container> --format '{{json .Mounts}}'`
mostró el mount real (`/data/JANUS-BCI-results → /results`, `RW: true`),
contra `docker compose config` mostrando la env var apuntando a otro lado —
la discrepancia entre ambos fue la señal.

**Lección:** cualquier variable de `.env` que se use tanto para el `source:`
de un volumen como para ser leída desde adentro del contenedor tiene dos
significados en conflicto. Si se agrega un volumen nuevo, declarar siempre
el path "de adentro" explícito en `environment:` — nunca asumir que el
valor de `.env` sirve tal cual del otro lado.

**Adenda (mismo día):** el fix de `docker-compose.yml` no alcanza para debuggear
desde VS Code — la extensión de Python carga `.env` por default en cualquier
sesión de debug (`python.envFile`, aplica aunque no esté explícito en
`.vscode/settings.json`), reintroduciendo el mismo conflicto de paths dentro
del proceso debuggeado. Fix: agregar un bloque `"env"` explícito a la
configuración en `.vscode/launch.json` con los paths de adentro del
contenedor — `"env"` se aplica después de cualquier `envFile` y gana.

## [2026-08-09] `runArgs` se ignora en `devcontainer.json` cuando se usa `dockerComposeFile`

**Contexto:** Armando `.devcontainer/devcontainer.json` para VS Code
Dev Containers, con `"runArgs": ["--gpus", "all"]` para pasar la GPU.

**Síntoma:** Ninguno todavía visible — se detectó revisando la
documentación oficial antes de que causara un fallo silencioso (la
GPU hubiera parecido no disponible dentro de VS Code sin ningún
error explicando por qué).

**Causa:** La spec de Dev Containers ignora por completo `runArgs`
cuando el devcontainer usa `dockerComposeFile` — ese campo solo
aplica al modo "single container" (`build`/`image` directo). Está
confirmado en la documentación oficial y en issues abiertos del repo
de Dev Containers que piden soporte para esto y todavía no lo tienen.

**Solución:** La GPU se configura en el `docker-compose.yml` mismo,
en el servicio que usa el devcontainer, con:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```
`devcontainer.json` solo necesita `"service": "bci-gpu"` apuntando a
ese servicio — nada de `runArgs`.

**Tags:** #docker #devcontainers #vscode #gpu

---

## [2026-08-06] docker compose run --rm es efímero — los cambios no persisten

**Contexto:** Probando si `setuptools==75.6.0` resolvía el error de
MLFlow, sin querer hacer un rebuild completo todavía.

**Síntoma:** Corrí `docker compose run --rm bci-gpu pip install setuptools==75.6.0`
y después `docker compose run --rm bci-gpu python scripts/verify_env.py`
en comandos separados — el segundo seguía fallando como si el
`pip install` nunca hubiera pasado.

**Causa:** Cada `docker compose run --rm` crea un contenedor nuevo
desde la imagen y lo borra al salir (`--rm`). Un `pip install` en
un contenedor no persiste al siguiente — solo persiste si se
reconstruye la imagen (`docker build`).

**Solución:** Para probar un fix sin rebuild, todo tiene que pasar
en el MISMO contenedor con un solo comando encadenado:
```bash
docker compose run --rm bci-gpu bash -c "pip install setuptools==75.6.0 && python scripts/verify_env.py"
```
Una vez confirmado que el fix funciona, recién ahí se actualiza el
`environment.yml` y se hace el rebuild real.

**Tags:** #docker #debugging

---

## [2026-08-06] `--gpus` no es un flag válido de `docker compose run`

**Contexto:** Intentando forzar acceso a GPU en un `run` puntual.

**Síntoma:**
```
docker compose --profile gpu run --rm --gpus all bci-gpu ...
unknown flag: --gpus
```

**Causa:** `--gpus` es un flag de `docker run` (Docker CLI puro),
no de `docker compose run`. Con Compose, el acceso a GPU se declara
en el `docker-compose.yml`, no en la línea de comandos.

**Solución:** En el servicio `bci-gpu` del `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```
Con esto, `docker compose run bci-gpu` ya tiene GPU sin flags extra.

**Tags:** #docker #gpu #compose

---

## [2026-08-06] Nombre de imagen impredecible (`janus-bci-bci-gpu`)

**Contexto:** Después de un build exitoso con `docker compose build`,
intenté correr la imagen con `docker run janus-bci` directo.

**Síntoma:**
```
docker: Error response from daemon: pull access denied for janus-bci,
repository does not exist or may require 'docker login'
```

**Causa:** Docker Compose nombra las imágenes automáticamente como
`<carpeta_del_proyecto>-<nombre_del_servicio>` — en este caso
`janus-bci-bci-gpu`, no `janus-bci`.

**Solución:** Agregar `image: janus-bci:gpu` explícito en el
servicio del `docker-compose.yml`. Así el nombre es predecible y
usable tanto con `docker compose run` como con `docker run` directo.

**Tags:** #docker #compose

---

## [2026-08-06] `pip install --index-url` dentro de `environment.yml` no funciona con mamba

**Contexto:** Intentando fijar `torch==2.3.0+cu118` desde el índice
oficial de PyTorch, directamente en el bloque `pip:` del yml.

**Síntoma:**
```
ERROR: Could not find a version that satisfies the requirement torch==2.3.0+cu118
(from versions: 1.13.0, ..., 2.13.0)
ERROR: No matching distribution found for torch==2.3.0+cu118
critical libmamba pip failed to install packages
```
Mamba ignoraba el `--index-url` de esa línea y buscaba en PyPI normal,
donde no existen los builds con sufijo `+cu118`.

**Causa:** El bloque `pip:` de un `environment.yml` no soporta bien
flags como `--index-url` por línea — mamba lo interpreta distinto a
un `pip install` de terminal.

**Solución:** Sacar PyTorch completamente del `environment.yml` y
instalarlo en una capa separada del `Dockerfile`, con un `pip install`
de terminal real:
```dockerfile
RUN pip install --no-cache-dir \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118
```
Ventaja extra: esta capa se cachea independiente del resto de
librerías — cambiar `mne` o `seaborn` no invalida el caché de PyTorch.

**Tags:** #docker #conda #pytorch

---

## [2026-08-06] MLFlow: `No module named 'pkg_resources'`

**Contexto:** Verificando el entorno con `verify_env.py` después de
que PyTorch, MNE y braindecode ya quedaron ✓.

**Síntoma:**
```
✗ MLFlow: No module named 'pkg_resources'
```

**Causa:** `setuptools` 83.0.0 (la versión más reciente en el momento)
ya no incluye `pkg_resources` — Python está deprecando ese módulo y
las versiones nuevas de setuptools lo sacaron. MLFlow todavía depende
de `pkg_resources` internamente.

**Solución:** Fijar una versión de `setuptools` anterior a la
deprecación, en el bloque `pip:` del `environment.yml`:
```yaml
- pip:
    - setuptools==75.6.0
```

**Tags:** #conda #mlflow #python

---

## [2026-08-06] `libstdc++.so.6: version 'CXXABI_1.3.15' not found`

**Contexto:** `mne-icalabel` y `braindecode` fallaban al importar,
ambos por la misma causa (el error apuntaba a `matplotlib` como
disparador, pero la causa raíz era la misma).

**Síntoma:**
```
✗ mne-icalabel: /lib/x86_64-linux-gnu/libstdc++.so.6: version
  `CXXABI_1.3.15' not found (required by .../matplotlib/_c_internal_utils...)
```

**Causa:** La imagen base de Ubuntu trae una versión de `libstdc++`
más vieja que la que necesitan las extensiones compiladas de algunas
librerías de Python instaladas por conda.

**Solución:** Instalar una versión más nueva de `libstdc++` DENTRO
del entorno conda (no del sistema), fijando la versión explícitamente
—sin versión fija, conda a veces no fuerza la actualización:
```yaml
- libstdcxx-ng=12
```

**Tags:** #conda #linux #compilación

---

## [2026-08-06] PyTorch instalado con versión/CUDA incorrectos vía conda

**Contexto:** Primer build completo con `environment.gpu.yml`
pidiendo `pytorch=2.3.0` y `pytorch-cuda=11.8` en los canales
`pytorch`/`nvidia`.

**Síntoma:**
```
✓ PyTorch: torch 2.13.0+cu130
```
Se instaló una versión mucho más nueva de la pedida, con CUDA 13.0
en vez de 11.8 — incompatible con la imagen base CUDA 11.8.

**Causa:** Conda/mamba resuelve versiones "compatibles" según todo
el árbol de dependencias del entorno, y en presencia de otras
librerías puede terminar eligiendo una versión de PyTorch distinta
a la pedida, ignorando el pin si hay conflictos de resolución.

**Solución:** Sacar PyTorch de conda por completo e instalarlo con
pip desde el índice oficial de PyTorch (ver entrada de
`--index-url` más arriba) — ahí las versiones son estrictas y no
hay reinterpretación del solver.

**Tags:** #conda #pytorch #gpu

---

## [2026-08-06] Mambaforge descontinuado — 404 en `docker build`

**Contexto:** Primera versión del `Dockerfile`, instalando conda
vía el instalador de Mambaforge.

**Síntoma:**
```
curl -fsSL .../Mambaforge-Linux-x86_64.sh -o /tmp/mambaforge.sh
curl: (22) The requested URL returned error: 404
```

**Causa:** El proyecto Mambaforge se fusionó con Miniforge — ya no
se publican releases nuevos de Mambaforge, la URL del "latest"
apunta a un release que no existe.

**Solución:** Usar Miniforge3 (el sucesor oficial, incluye mamba
igual):
```
Miniforge3-Linux-x86_64.sh   # en vez de Mambaforge-Linux-x86_64.sh
```

**Tags:** #docker #conda

---

## [2026-08-06] Primer `docker build` corrió con el Dockerfile viejo

**Contexto:** Reemplacé el `Dockerfile` y creé `environment.yml`,
pero el build seguía sin encontrar `environment.yml`.

**Síntoma:**
```
cat environment.yml
No such file or directory
```
Y el log del build mostraba `FROM python:3.11-slim` — el Dockerfile
genérico original, no el nuevo con CUDA + conda.

**Causa:** El archivo nunca se había reemplazado en disco — el
`cat > Dockerfile << EOF ... EOF` no se había ejecutado todavía en
esa sesión de terminal.

**Solución:** Verificar SIEMPRE el contenido real en disco antes de
un build largo:
```bash
head -3 Dockerfile
```
Evita perder 5+ minutos de build con el archivo equivocado.

**Tags:** #docker #debugging

---

## [2026-08-06] `docker: permission denied` al conectar al socket

**Contexto:** Primer `docker run` después de instalar Docker y el
NVIDIA Container Toolkit.

**Síntoma:**
```
permission denied while trying to connect to the docker API at
unix:///var/run/docker.sock
```

**Causa:** El usuario no pertenece al grupo `docker` — sin eso,
todo comando Docker requiere `sudo`.

**Solución:**
```bash
sudo usermod -aG docker $USER
newgrp docker   # aplica el grupo en la sesión actual sin logout
```
Para que quede permanente en todas las terminales nuevas, cerrar
sesión y volver a entrar.

**Tags:** #docker #linux #permisos

---

## [2026-08-06] Repositorio de NVIDIA con `$(ARCH)` sin expandir

**Contexto:** Después de agregar el repo de NVIDIA para el
Container Toolkit.

**Síntoma:**
```
E: Unable to locate package nvidia-container-toolkit
```
El archivo `/etc/apt/sources.list.d/nvidia-container-toolkit.list`
tenía literalmente `$(ARCH)` como texto, sin expandir a `amd64`.

**Causa:** El comando oficial de NVIDIA usa una sustitución de shell
(`$(dpkg --print-architecture)`) que en algunos shells/contextos
(ej. Zorin OS con ciertos alias) no se expande como se espera.

**Solución:** Escribir el archivo con la arquitectura literal:
```bash
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
```

**Tags:** #linux #nvidia #apt

---

## [2026-08-06] `apt update` bloqueado por repos de terceros rotos

**Contexto:** Primer intento de instalar `nvidia-container-toolkit`.

**Síntoma:**
```
E: The repository 'https://microsoft.com stable Release' does not have a Release file.
E: The repository 'https://ppa.launchpadcontent.net/gezakovacs/ppa/ubuntu jammy Release' does not have a Release file.
```
Estos errores no bloquean `apt install` de paquetes ya conocidos,
pero si el paquete todavía no está en el índice local, `apt update`
tiene que completarse sin errores fatales primero.

**Causa:** Dos repos de terceros preexistentes en el sistema
(instalados por otro software, no relacionados a JANUS) estaban
rotos — sin archivo `Release` válido.

**Solución:** Deshabilitarlos temporalmente (no borrarlos, por si
se usan para otra cosa):
```bash
sudo mv /etc/apt/sources.list.d/vscode.list /etc/apt/sources.list.d/vscode.list.bak
sudo mv /etc/apt/sources.list.d/gezakovacs-ubuntu-ppa-jammy.list /etc/apt/sources.list.d/gezakovacs-ubuntu-ppa-jammy.list.bak
```

**Tags:** #linux #apt

"""
Punto de entrada único para una corrida de producción (un barrido:
todas las partitions de un recipe+dataset). Esto es lo que lo
diferencia de sandbox/: todo lo que pasa por acá queda trackeado.

Responsabilidades (ver PROTOCOL.md secciones 5 y 7):
    1. Recibir un config.json.
    2. Loguear el trío de reproducibilidad (config.json, .git_commit,
       .docker_image) en cada carpeta hoja.
    3. Escribir/actualizar script_progress.csv a nivel del barrido.
    4. Por cada partition: delegar el training/analysis real a
       src/training o src/analysis, y dejar metrics_results.csv en la
       hoja correspondiente, con el contrato exacto de PROTOCOL.md
       sección 6.

Es un esqueleto: la lógica específica de tu dominio (BCI, imágenes,
PINNs) vive en src/, no acá. Este archivo solo orquesta y garantiza
que el contrato se cumpla siempre igual, sin importar el dominio.
"""
import argparse
import json
import subprocess
from pathlib import Path

from src.utils.paths import PATHS


def get_git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def make_run_dir(strategy: str, recipe: str, dataset: str, partition: str, replicate: str) -> Path:
    run_dir = PATHS.results_root / strategy / recipe / dataset / partition / replicate
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def log_reproducibility_trio(run_dir: Path, config: dict, docker_image: str | None):
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / ".git_commit").write_text(get_git_commit_hash())
    if docker_image:
        (run_dir / ".docker_image").write_text(docker_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path a config.json")
    parser.add_argument("--docker-image", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    # TODO: iterar sobre context.partitions definidas en config,
    # llamar a la lógica real de src/training o src/analysis por cada
    # una, escribir script_progress.csv (una fila por partition) y
    # metrics_results.csv (una fila por métrica) siguiendo el
    # contrato de PROTOCOL.md sección 6.
    print("Esqueleto de run_production.py — completar con tu lógica de dominio.")
    print(f"Config recibida: {config.get('project_name', '???')} / "
          f"{config.get('strategy_name', '???')} / {config.get('recipe_name', '???')}")


if __name__ == "__main__":
    main()

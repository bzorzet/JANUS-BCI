"""
Trío de reproducibilidad radical (PROTOCOL.md sección 8): config.json,
.git_commit, .docker_image. Compartido entre scripts/run_production.py
y src/preprocessing/database_preprocessor.py — no duplicar esta lógica.
"""
import json
import subprocess
from pathlib import Path
from typing import Optional


def get_git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def log_reproducibility_trio(run_dir: Path, config: dict, docker_image: Optional[str] = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / ".git_commit").write_text(get_git_commit_hash())
    if docker_image:
        (run_dir / ".docker_image").write_text(docker_image)

import pytest
from pydantic import ValidationError

from src.utils.paths import JanusPaths


def test_paths_fails_fast_on_nonexistent_directory(tmp_path):
    missing_dir = tmp_path / "no-existe"

    with pytest.raises(ValidationError):
        JanusPaths(
            _env_file=None,
            janus_repo_root=missing_dir,
            janus_data_root=tmp_path,
            janus_results_root=tmp_path / "results",
            janus_sandbox_root=tmp_path / "sandbox",
        )


def test_paths_succeeds_when_required_dirs_exist(tmp_path):
    paths = JanusPaths(
        _env_file=None,
        janus_repo_root=tmp_path,
        janus_data_root=tmp_path,
        janus_results_root=tmp_path / "results",
        janus_sandbox_root=tmp_path / "sandbox",
    )

    assert paths.data_root == tmp_path
    assert paths.results_root.exists()
    assert paths.sandbox_root.exists()

import pandas as pd
import pytest

from src.training.weights.weights_matcher import WithinSubjectMatcher
from src.training.weights.weights_resolver import TrainedModelResolver


def _make_training_tree(results_root, strategy="WS-Standard", recipe="CTNet", dataset="Cho2017",
                         partitions_status=None):
    """partitions_status: dict subject_partition -> status. Crea
    subject_XX/seed_N/best_model.pth para cada partición y un
    script_progress.csv con esos status."""
    partitions_status = partitions_status or {}
    root = results_root / strategy / recipe / dataset
    root.mkdir(parents=True)

    rows = []
    for partition, status in partitions_status.items():
        for seed in (1, 2):
            leaf = root / partition / f"seed_{seed}"
            leaf.mkdir(parents=True)
            (leaf / "best_model.pth").write_bytes(b"fake-weights")
            # a stray extra .pth that should NOT be picked up (checkpoint,
            # not best_model.pth)
            (leaf / "params_epoch_0.pth").write_bytes(b"fake-checkpoint")
        rows.append({"partition": partition, "status": status,
                     "timestamp_start": "t0", "timestamp_end": "t1"})
    pd.DataFrame(rows).to_csv(root / "script_progress.csv", index=False)
    return root


def test_resolver_only_returns_best_model_pth(tmp_path):
    _make_training_tree(tmp_path, partitions_status={"subject_01": "success"})
    resolver = TrainedModelResolver(
        tmp_path, {"strategy_name": "WS-Standard", "recipe_name": "CTNet", "database_name": "Cho2017"},
    )
    weights = list(resolver.resolve_weights())
    assert len(weights) == 2  # seed_1, seed_2 -- not params_epoch_0.pth
    for w in weights:
        assert w.path.name == "best_model.pth"


def test_resolver_excludes_failed_partitions_even_if_file_exists(tmp_path):
    _make_training_tree(tmp_path, partitions_status={"subject_01": "success", "subject_02": "failed"})
    resolver = TrainedModelResolver(
        tmp_path, {"strategy_name": "WS-Standard", "recipe_name": "CTNet", "database_name": "Cho2017"},
    )
    weights = list(resolver.resolve_weights())
    partitions = {w.metadata["segments"][0] for w in weights}
    assert partitions == {"subject_01"}


def test_resolver_scoped_to_strategy_recipe_dataset(tmp_path):
    _make_training_tree(tmp_path, strategy="WS-Standard", recipe="CTNet", dataset="Cho2017",
                         partitions_status={"subject_01": "success"})
    _make_training_tree(tmp_path, strategy="WS-Standard", recipe="EEGNetv4", dataset="Cho2017",
                         partitions_status={"subject_01": "success"})

    resolver = TrainedModelResolver(
        tmp_path, {"strategy_name": "WS-Standard", "recipe_name": "CTNet", "database_name": "Cho2017"},
    )
    weights = list(resolver.resolve_weights())
    assert all("CTNet" in str(w.path) for w in weights)
    assert not any("EEGNetv4" in str(w.path) for w in weights)


def test_resolver_missing_root_yields_nothing(tmp_path):
    resolver = TrainedModelResolver(
        tmp_path, {"strategy_name": "Nope", "recipe_name": "Nope", "database_name": "Nope"},
    )
    assert list(resolver.resolve_weights()) == []


def test_within_subject_matcher_matches_all_seeds_of_subject(tmp_path):
    _make_training_tree(tmp_path, partitions_status={"subject_01": "success", "subject_02": "success"})
    resolver = TrainedModelResolver(
        tmp_path, {"strategy_name": "WS-Standard", "recipe_name": "CTNet", "database_name": "Cho2017"},
    )
    all_weights = list(resolver.resolve_weights())
    matcher = WithinSubjectMatcher()

    matched = matcher.match(all_weights, "subject_01")
    assert len(matched) == 2
    assert all(w.metadata["segments"][0] == "subject_01" for w in matched)

    matched_none = matcher.match(all_weights, "subject_99")
    assert matched_none == []

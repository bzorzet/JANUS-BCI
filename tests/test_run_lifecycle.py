import json

import pandas as pd
import pytest

from src.utils.run_lifecycle import RunLifecycleManager, _diff_values


def _write_identity(output_path, identity):
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "config.json").write_text(json.dumps(identity))


def _write_progress(output_path, rows, filename="script_progress.csv"):
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path / filename, index=False)


# --- check_identity_or_raise -------------------------------------------------

def test_identity_matches_no_raise(tmp_path):
    _write_identity(tmp_path, {"pipeline": {"a": 1}})
    lifecycle = RunLifecycleManager(tmp_path)
    lifecycle.check_identity_or_raise({"pipeline": {"a": 1}})  # no raise


def test_identity_missing_config_is_first_run(tmp_path):
    lifecycle = RunLifecycleManager(tmp_path)
    lifecycle.check_identity_or_raise({"pipeline": {"a": 1}})  # no raise, nothing to compare


def test_identity_mismatch_no_tty_rejects_hard(tmp_path, monkeypatch):
    _write_identity(tmp_path, {"pipeline": {"a": 1}})
    lifecycle = RunLifecycleManager(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(RuntimeError):
        lifecycle.check_identity_or_raise({"pipeline": {"a": 2}})


def test_identity_mismatch_tty_accept_resets(tmp_path, monkeypatch):
    _write_identity(tmp_path, {"pipeline": {"a": 1}})
    _write_progress(tmp_path, [{"partition": "s01", "status": "success",
                                 "timestamp_start": "t0", "timestamp_end": "t1"}])
    lifecycle = RunLifecycleManager(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    lifecycle.check_identity_or_raise({"pipeline": {"a": 2}})
    assert not lifecycle.progress_path.exists()


def test_identity_mismatch_tty_reject_raises(tmp_path, monkeypatch):
    _write_identity(tmp_path, {"pipeline": {"a": 1}})
    lifecycle = RunLifecycleManager(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    with pytest.raises(RuntimeError):
        lifecycle.check_identity_or_raise({"pipeline": {"a": 2}})


def test_identity_mismatch_overwrite_existing_resets_without_prompt(tmp_path, monkeypatch):
    _write_identity(tmp_path, {"pipeline": {"a": 1}})
    _write_progress(tmp_path, [{"partition": "s01", "status": "success",
                                 "timestamp_start": "t0", "timestamp_end": "t1"}])
    lifecycle = RunLifecycleManager(tmp_path, overwrite_existing=True)
    called = {"prompted": False}
    monkeypatch.setattr("sys.stdin.isatty", lambda: (_ for _ in ()).throw(AssertionError("should not check TTY")))
    lifecycle.check_identity_or_raise({"pipeline": {"a": 2}})
    assert not lifecycle.progress_path.exists()


def test_identity_mismatch_extra_cleanup_runs_on_overwrite(tmp_path):
    _write_identity(tmp_path, {"pipeline": {"a": 1}})
    extra_file = tmp_path / "stage_metrics.csv"
    extra_file.write_text("x")
    lifecycle = RunLifecycleManager(tmp_path, overwrite_existing=True)
    lifecycle.check_identity_or_raise(
        {"pipeline": {"a": 2}},
        extra_cleanup=lambda: extra_file.unlink(),
    )
    assert not extra_file.exists()


# --- resume: load_success_partitions / filter_pending / prepare_partitions --

def test_load_success_partitions_filters_by_status_and_presence(tmp_path):
    _write_progress(tmp_path, [
        {"partition": "s01", "status": "success", "timestamp_start": "t0", "timestamp_end": "t1"},
        {"partition": "s02", "status": "failed", "timestamp_start": "t0", "timestamp_end": "t1"},
        {"partition": "s03", "status": "success", "timestamp_start": "t0", "timestamp_end": "t1"},
    ])
    lifecycle = RunLifecycleManager(tmp_path)
    present = {"s01"}  # s03 marked success but its output was deleted out-of-band
    done = lifecycle.load_success_partitions(still_present=lambda p: p in present)
    assert done == {"s01"}


def test_load_success_partitions_keeps_latest_duplicate_row(tmp_path):
    _write_progress(tmp_path, [
        {"partition": "s01", "status": "failed", "timestamp_start": "t0", "timestamp_end": "t1"},
        {"partition": "s01", "status": "success", "timestamp_start": "t2", "timestamp_end": "t3"},
    ])
    lifecycle = RunLifecycleManager(tmp_path)
    done = lifecycle.load_success_partitions(still_present=lambda p: True)
    assert done == {"s01"}


def test_load_success_partitions_no_progress_file(tmp_path):
    lifecycle = RunLifecycleManager(tmp_path)
    assert lifecycle.load_success_partitions(still_present=lambda p: True) == set()


def test_filter_pending_drops_done():
    lifecycle = RunLifecycleManager
    manager = lifecycle.__new__(lifecycle)  # filter_pending needs no instance state
    result = RunLifecycleManager.filter_pending(manager, ["s01", "s02", "s03"], {"s02"})
    assert result == ["s01", "s03"]


def test_filter_pending_with_key_extractor():
    manager = RunLifecycleManager.__new__(RunLifecycleManager)
    items = [(1, "a", "s01"), (1, "b", "s02")]
    result = RunLifecycleManager.filter_pending(manager, items, {"s01"}, key=lambda item: item[2])
    assert result == [(1, "b", "s02")]


def test_prepare_partitions_force_skips_resume(tmp_path):
    _write_progress(tmp_path, [
        {"partition": "s01", "status": "success", "timestamp_start": "t0", "timestamp_end": "t1"},
    ])
    lifecycle = RunLifecycleManager(tmp_path, force=True)
    pending = lifecycle.prepare_partitions(["s01", "s02"], still_present=lambda p: True)
    assert pending == ["s01", "s02"]


def test_prepare_partitions_without_force_excludes_done(tmp_path):
    _write_progress(tmp_path, [
        {"partition": "s01", "status": "success", "timestamp_start": "t0", "timestamp_end": "t1"},
    ])
    lifecycle = RunLifecycleManager(tmp_path, force=False)
    pending = lifecycle.prepare_partitions(["s01", "s02"], still_present=lambda p: True)
    assert pending == ["s02"]


# --- progress writes ---------------------------------------------------------

def test_mark_start_default_emits_no_row(tmp_path):
    lifecycle = RunLifecycleManager(tmp_path)
    ts = lifecycle.mark_start("s01")
    assert ts
    assert not lifecycle.progress_path.exists()


def test_mark_start_emit_running_row(tmp_path):
    lifecycle = RunLifecycleManager(tmp_path, emit_running_row=True)
    lifecycle.mark_start("s01")
    df = pd.read_csv(lifecycle.progress_path)
    assert list(df["status"]) == ["running"]


def test_mark_result_appends_row_with_protocol_columns(tmp_path):
    lifecycle = RunLifecycleManager(tmp_path)
    ts_start = lifecycle.mark_start("s01")
    lifecycle.mark_result("s01", "success", ts_start)
    df = pd.read_csv(lifecycle.progress_path)
    assert list(df.columns) == ["partition", "status", "timestamp_start", "timestamp_end"]
    assert df.iloc[0]["partition"] == "s01"
    assert df.iloc[0]["status"] == "success"


# --- _diff_values --------------------------------------------------------

def test_diff_values_reports_added_removed_changed_keys():
    old = {"a": 1, "b": 2}
    new = {"a": 1, "b": 3, "c": 4}
    diffs = _diff_values(old, new)
    joined = "\n".join(diffs)
    assert "c" in joined and "agregado" in joined
    assert "identity.b" in joined

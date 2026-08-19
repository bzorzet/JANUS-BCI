import numpy as np
import pytest

from src.training.splitting.splitters import WithinSubjectHoldoutSplitter, stratified_sequential_split


def test_stratified_sequential_split_is_deterministic_and_floor_divides():
    labels = np.array([0, 1] * 10)  # 20 samples, balanced
    train_idx, test_idx = stratified_sequential_split(ntest=4, labels=labels)
    # ntest=4 over 2 classes -> 2 per class; ntrain=16 over 2 classes -> 8 per class
    assert len(test_idx) == 4
    assert len(train_idx) == 16
    # deterministic: same call twice, same result
    train_idx2, test_idx2 = stratified_sequential_split(ntest=4, labels=labels)
    assert train_idx == train_idx2
    assert test_idx == test_idx2


def test_stratified_sequential_split_drops_remainder_on_uneven_division():
    # 3 classes, 5 samples each (15 total). ntest=4 doesn't divide evenly
    # by 3 classes (floor(4/3)=1 per class), and ntrain=11 doesn't either
    # (floor(11/3)=3 per class) -- 1 sample per class (3 total) is silently
    # dropped from both train and test. Preserved verbatim from repo_viejo,
    # not a bug to fix in this refactor.
    labels = np.array([0, 1, 2] * 5)
    train_idx, test_idx = stratified_sequential_split(ntest=4, labels=labels)
    assert len(train_idx) == 9  # 3 per class
    assert len(test_idx) == 3   # 1 per class
    assert len(train_idx) + len(test_idx) < len(labels)  # 3 samples dropped


def test_within_subject_holdout_splitter_respects_proportions():
    n = 100
    X = np.arange(n).reshape(n, 1)
    y = np.array([0, 1] * (n // 2))

    splitter = WithinSubjectHoldoutSplitter(ptrain=0.6, pval=0.2, ptest=0.2)
    fold = splitter.split(X, y, metadata=None, subject_id="subject_08")

    assert len(fold.X_test) == 20
    assert len(fold.X_val) == 20
    assert len(fold.X_train) == 60
    assert fold.metadata["subject_id"] == "subject_08"
    assert len(fold.metadata["test_idx"]) == 20
    assert len(fold.metadata["val_idx"]) == 20
    assert len(fold.metadata["train_idx"]) == 60


def test_within_subject_holdout_splitter_is_deterministic_regardless_of_seed():
    # No seed parameter exists at all -- this is intentional (decisión ya
    # cerrada con el usuario, ver splitters.py docstring). Two independent
    # splitters with identical ptrain/pval/ptest always agree.
    n = 60
    X = np.arange(n).reshape(n, 1)
    y = np.array([0, 1] * (n // 2))

    splitter_a = WithinSubjectHoldoutSplitter(ptrain=0.6, pval=0.2, ptest=0.2)
    splitter_b = WithinSubjectHoldoutSplitter(ptrain=0.6, pval=0.2, ptest=0.2)

    fold_a = splitter_a.split(X, y, metadata=None, subject_id=1)
    fold_b = splitter_b.split(X, y, metadata=None, subject_id=1)

    assert np.array_equal(fold_a.X_train, fold_b.X_train)
    assert np.array_equal(fold_a.X_val, fold_b.X_val)
    assert np.array_equal(fold_a.X_test, fold_b.X_test)
    assert fold_a.metadata["train_idx"] == fold_b.metadata["train_idx"]

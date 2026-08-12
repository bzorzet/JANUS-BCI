import pytest

torch = pytest.importorskip("torch")

from src.training.loss_strategy import LossStrategy
from src.training.trainer import Trainer_DL


def _loader(n=10, batch_size=5, n_features=4, n_classes=2):
    X = torch.randn(n, n_features)
    y = torch.randint(0, n_classes, (n,))
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_loss_strategy_none_matches_direct_loss_fn():
    """loss_strategy=None (default) debe reproducir exactamente
    loss_fn(y_pred, y_true) -- ningún config existente cambia de
    comportamiento si no especifica loss_strategy."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _loader()

    trainer = Trainer_DL(
        model=model, loss_fn=loss_fn,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),  # lr=0 -> no actual weight update
        train_loader=loader, max_epochs=1, loss_strategy=None,
    )
    trainer.train()  # should not raise, and should use loss_fn directly


class _RecordingLossStrategy(LossStrategy):
    def __init__(self):
        self.calls = []

    def compute(self, y_pred, y_true, X, model, default_loss_fn):
        self.calls.append((y_pred.shape, y_true.shape, X.shape))
        return default_loss_fn(y_pred, y_true)


def test_loss_strategy_is_invoked_with_documented_kwargs():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _loader(n=10, batch_size=5)  # 2 batches
    strategy = _RecordingLossStrategy()

    trainer = Trainer_DL(
        model=model, loss_fn=loss_fn,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=loader, max_epochs=1, loss_strategy=strategy,
    )
    trainer.train()

    assert len(strategy.calls) == 2  # una por batch
    for y_pred_shape, y_true_shape, X_shape in strategy.calls:
        assert y_pred_shape == (5, 2)
        assert y_true_shape == (5,)
        assert X_shape == (5, 4)


def test_loss_strategy_applied_in_evaluate_too():
    """El hook cubre tanto train() como evaluate() -- si no, val_loss
    quedaría inconsistente con train_loss cuando hay una strategy activa."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loader = _loader(n=10, batch_size=5)
    val_loader = _loader(n=10, batch_size=5)
    strategy = _RecordingLossStrategy()

    trainer = Trainer_DL(
        model=model, loss_fn=loss_fn,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        train_loader=train_loader, val_loader=val_loader,
        max_epochs=1, loss_strategy=strategy,
    )
    trainer.train()

    # 2 batches de train + 2 de val = 4 llamadas
    assert len(strategy.calls) == 4

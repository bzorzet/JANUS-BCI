import pytest

torch = pytest.importorskip("torch")

from src.training.tester import Tester_DL


class _CountingLinear(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.base(x)


def _loader(n=20, batch_size=5, n_features=4, n_classes=2):
    X = torch.randn(n, n_features)
    y = torch.randint(0, n_classes, (n,))
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_model_infer_called_exactly_once_per_batch():
    """Regresión del bug ya aprobado: el Tester original llamaba
    model_infer(X) DOS veces por batch (cómputo redundante). Acá debe
    llamarse exactamente una vez por batch."""
    loader = _loader(n=20, batch_size=5)  # 4 batches
    base_model = torch.nn.Linear(4, 2)
    model = _CountingLinear(base_model)

    tester = Tester_DL(model, loss_fn=torch.nn.CrossEntropyLoss())
    y_pred, loss = tester.test(loader)

    assert model.calls == 4
    assert y_pred.shape == (20, 2)
    assert loss is not None


def test_probability_false_returns_argmax():
    loader = _loader(n=10, batch_size=10)
    model = torch.nn.Linear(4, 2)
    tester = Tester_DL(model, loss_fn=None)

    y_pred, loss = tester.test(loader, probability=False)

    assert y_pred.shape == (10,)
    assert loss is None


def test_no_loss_fn_returns_none_loss():
    loader = _loader(n=8, batch_size=4)
    model = torch.nn.Linear(4, 2)
    tester = Tester_DL(model, loss_fn=None)

    _, loss = tester.test(loader, probability=True)

    assert loss is None


def test_model_infer_supports_dict_style_batches():
    class DictModel(torch.nn.Module):
        def forward(self, X, A):
            return X.sum(dim=-1, keepdim=True).expand(-1, 2)

    model = DictModel()
    tester = Tester_DL(model)
    X = {"X": torch.randn(3, 4), "A": torch.randn(4, 4)}
    y_pred = tester.model_infer(X)
    assert y_pred.shape == (3, 2)

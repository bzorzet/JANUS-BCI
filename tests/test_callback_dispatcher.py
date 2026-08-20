from src.training.core.callbacks.callback_dispatcher import CallbackDispatcher


class _FullCallback:
    def __init__(self):
        self.calls = []

    def on_train_begin(self, trainer, **kwargs):
        self.calls.append(("on_train_begin", trainer))

    def on_epoch_end(self, trainer, epoch, **kwargs):
        self.calls.append(("on_epoch_end", trainer, epoch))


class _PartialCallback:
    """Solo implementa on_batch_end -- no debe registrarse en ningún otro evento."""
    def __init__(self):
        self.calls = []

    def on_batch_end(self, trainer, batch_idx, **kwargs):
        self.calls.append(("on_batch_end", trainer, batch_idx))


def test_only_callbacks_implementing_event_get_registered():
    full_cb = _FullCallback()
    partial_cb = _PartialCallback()
    dispatcher = CallbackDispatcher([full_cb, partial_cb])

    assert full_cb in dispatcher.callbacks["on_train_begin"]
    assert full_cb in dispatcher.callbacks["on_epoch_end"]
    assert full_cb not in dispatcher.callbacks["on_batch_end"]

    assert partial_cb in dispatcher.callbacks["on_batch_end"]
    assert partial_cb not in dispatcher.callbacks["on_train_begin"]
    assert partial_cb not in dispatcher.callbacks["on_epoch_end"]


def test_notify_passes_trainer_positionally_and_kwargs():
    cb = _FullCallback()
    dispatcher = CallbackDispatcher([cb])
    sentinel_trainer = object()

    dispatcher.notify("on_epoch_end", trainer=sentinel_trainer, epoch=5)

    assert cb.calls == [("on_epoch_end", sentinel_trainer, 5)]


def test_notify_on_event_with_no_registered_callbacks_is_noop():
    dispatcher = CallbackDispatcher([])
    dispatcher.notify("on_train_end", trainer=object())  # should not raise


def test_empty_callbacks_list_default():
    dispatcher = CallbackDispatcher()
    for event in CallbackDispatcher.EVENTS:
        assert dispatcher.callbacks[event] == []

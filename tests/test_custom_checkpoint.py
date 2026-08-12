from src.torch_utils import CustomCheckpoint


def test_save_best_param_matches_real_config_key(tmp_path):
    """Regresión del bug ya aprobado: el config real usa la clave
    `save_best` (no `save_best_model`) -- antes del fix, caía
    silenciosamente en **kwargs y save_best_model quedaba siempre True sin
    importar el config."""
    cb = CustomCheckpoint(dirname=str(tmp_path), save_each_n_epochs=100, save_best=True)
    assert cb.save_best_model is True


def test_save_best_false_is_respected(tmp_path):
    cb = CustomCheckpoint(dirname=str(tmp_path), save_each_n_epochs=100, save_best=False)
    assert cb.save_best_model is False


def test_default_save_best_is_true(tmp_path):
    cb = CustomCheckpoint(dirname=str(tmp_path))
    assert cb.save_best_model is True

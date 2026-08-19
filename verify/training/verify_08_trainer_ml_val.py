"""
Script de verificación -- Trainer_ML con X_val/y_val opcionales (preparado
para selección de modelo a futuro, sin uso todavía dentro de la clase).

Corré esto a mano (modo debug), no vía pytest.

Qué verifica:
1. Compatibilidad hacia atrás: instanciar SIN X_val/y_val (como hacía
   cualquier caller antes de este cambio) sigue funcionando exactamente
   igual -- train()/infer()/get_history() sin ningún cambio de
   comportamiento.
2. Con X_val/y_val SÍ pasados, quedan guardados como atributos accesibles
   (aunque la clase no los use todavía internamente) -- confirma que el
   dato no se pierde, está disponible para cuando se implemente selección.
3. get_history() sigue devolviendo {} en ambos casos (con y sin val) --
   no se filtró ningún efecto secundario nuevo.
4. train() e infer() dan el MISMO resultado con o sin val pasado (val no
   debe alterar el entrenamiento/inferencia todavía, ya que no se usa).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.training.core.trainer import Trainer_ML


def _build_dummy_classification_data(n=40, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # separable de forma simple, para que el modelo converja bien
    return X, y


def verify_backward_compatible_without_val():
    print("--- 1. Compatibilidad hacia atrás: instanciar SIN X_val/y_val funciona igual que antes ---")
    X_train, y_train = _build_dummy_classification_data()
    model = LogisticRegression()
    trainer = Trainer_ML(model, X_train, y_train)  # firma vieja, sin val

    assert trainer.X_val is None, f"X_val debería ser None por default, es {trainer.X_val}"
    assert trainer.y_val is None, f"y_val debería ser None por default, es {trainer.y_val}"

    trainer.train()
    y_pred = trainer.infer(X_train, probability=False)
    assert y_pred.shape == y_train.shape, "infer() debería devolver un array del mismo shape que y_train"
    assert trainer.get_history() == {}, "get_history() debería seguir devolviendo {} (sin cambios)"

    print(f"  X_val={trainer.X_val}, y_val={trainer.y_val} (default None, como antes del cambio)")
    print(f"  accuracy en train (sanity check): {(y_pred == y_train).mean():.4f}")
    print("  OK -- comportamiento idéntico al de antes de agregar X_val/y_val\n")


def verify_val_is_stored_when_provided():
    print("--- 2. Con X_val/y_val pasados, quedan guardados y accesibles (aunque no se usen todavía) ---")
    X_train, y_train = _build_dummy_classification_data(n=40, seed=0)
    X_val, y_val = _build_dummy_classification_data(n=10, seed=1)

    model = LogisticRegression()
    trainer = Trainer_ML(model, X_train, y_train, X_val=X_val, y_val=y_val)

    assert trainer.X_val is X_val, "trainer.X_val debería ser exactamente el array pasado (misma identidad)"
    assert trainer.y_val is y_val, "trainer.y_val debería ser exactamente el array pasado (misma identidad)"
    print(f"  trainer.X_val.shape = {trainer.X_val.shape}, trainer.y_val.shape = {trainer.y_val.shape}")
    print("  OK -- val queda disponible como atributo para uso futuro\n")


def verify_val_does_not_affect_training_yet():
    print("--- 3 y 4. val NO altera train()/infer() todavía (no se usa internamente) ---")
    X_train, y_train = _build_dummy_classification_data(n=40, seed=0)
    X_val, y_val = _build_dummy_classification_data(n=10, seed=1)

    # Mismo modelo (misma clase, mismos hiperparámetros default), mismo
    # X_train/y_train -- la ÚNICA diferencia entre trainer_a y trainer_b es
    # si se le pasó val o no. Si algo internamente usara val hoy (no
    # debería), los resultados divergirían.
    model_a = LogisticRegression()
    trainer_a = Trainer_ML(model_a, X_train, y_train)  # sin val
    trainer_a.train()
    pred_a = trainer_a.infer(X_train, probability=False)

    model_b = LogisticRegression()
    trainer_b = Trainer_ML(model_b, X_train, y_train, X_val=X_val, y_val=y_val)  # con val
    trainer_b.train()
    pred_b = trainer_b.infer(X_train, probability=False)

    assert np.array_equal(pred_a, pred_b), (
        "Las predicciones difieren entre el Trainer_ML sin val y el que SÍ recibió val -- "
        "val está afectando el entrenamiento/inferencia, cuando todavía no debería (no está "
        "implementada ninguna lógica de selección que lo use)."
    )
    assert trainer_a.get_history() == trainer_b.get_history() == {}, "get_history() debería ser {} en ambos casos"

    print(f"  predicciones idénticas con y sin val pasado ({len(pred_a)} muestras)")
    print("  OK -- val no tiene ningún efecto todavía, como se espera en este estado del contrato\n")


if __name__ == "__main__":
    verify_backward_compatible_without_val()
    verify_val_is_stored_when_provided()
    verify_val_does_not_affect_training_yet()
    print("=== TODOS LOS CHECKS DE verify_08_trainer_ml_val.py PASARON ===")

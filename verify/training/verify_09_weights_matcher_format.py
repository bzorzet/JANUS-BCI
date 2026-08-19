"""
Script de verificación -- WithinSubjectMatcher con validación de formato
(falla fuerte ante partition con formato inesperado, en vez de devolver
lista vacía en silencio).

Corré esto a mano (modo debug), no vía pytest.

Qué verifica:
1. Caso normal: test_partition y TrainedWeights con formato subject_XX
   matchean correctamente, sin ningún cambio de comportamiento respecto a
   antes de agregar la validación.
2. Caso normal: sujeto sin ningún TrainedWeights correspondiente devuelve
   lista vacía (esto SIGUE siendo válido -- "no hay pesos para este
   sujeto" es una respuesta legítima, distinta de "formato irreconocible").
3. test_partition con formato inesperado (ej. viniera de LeaveOneSubjectOut)
   -> debe hacer raise ValueError, no devolver lista vacía en silencio.
4. TrainedWeights con partition de origen de formato inesperado -> debe
   hacer raise ValueError también (no solo se valida el lado de destino).
5. El mensaje de error menciona explícitamente qué se recibió y por qué
   falló -- no un error genérico sin contexto.
"""
from pathlib import Path

from src.training.weights.matcher import WithinSubjectMatcher
from src.training.weights.resolver import TrainedWeights


def _build_weights(partition_str: str, path_str: str = "dummy.pth") -> TrainedWeights:
    """Construye un TrainedWeights dummy con metadata['segments'] derivado
    de partition_str, igual que hace TrainedModelResolver.resolve_weights()
    real (segments = rel.parts[:-1])."""
    segments = tuple(partition_str.split("/"))
    return TrainedWeights(path=Path(path_str), partition=partition_str, metadata={"segments": segments})


def verify_normal_case_matches_correctly():
    print("--- 1. Caso normal: subject_XX matchea correctamente ---")
    matcher = WithinSubjectMatcher()
    trained_weights = [
        _build_weights("subject_08/split_8_seed_399"),
        _build_weights("subject_08/split_8_seed_100"),
        _build_weights("subject_09/split_8_seed_399"),
    ]

    matched = matcher.match(trained_weights, "subject_08")
    assert len(matched) == 2, f"Se esperaban 2 TrainedWeights para subject_08, se obtuvieron {len(matched)}"
    assert all(w.metadata["segments"][0] == "subject_08" for w in matched), "Todos los matched deberían ser de subject_08"
    print(f"  matched: {[w.partition for w in matched]}")
    print("  OK\n")


def verify_no_match_returns_empty_list():
    print("--- 2. Sujeto sin pesos correspondientes -> lista vacía (sigue siendo válido) ---")
    matcher = WithinSubjectMatcher()
    trained_weights = [_build_weights("subject_08/split_8_seed_399")]

    matched = matcher.match(trained_weights, "subject_15")  # subject_15 no tiene pesos, pero formato válido
    assert matched == [], f"Se esperaba lista vacía para subject_15 (sin pesos), se obtuvo {matched}"
    print("  subject_15 (formato válido, sin pesos entrenados) -> [] , sin excepción")
    print("  OK\n")


def verify_unexpected_test_partition_format_raises():
    print("--- 3. test_partition con formato inesperado -> ValueError (falla fuerte) ---")
    matcher = WithinSubjectMatcher()
    trained_weights = [_build_weights("subject_08/split_8_seed_399")]

    try:
        matcher.match(trained_weights, "loso_test-subject_08")  # formato LOSO, no subject_XX
        raised = False
    except ValueError as e:
        raised = True
        error_message = str(e)

    assert raised, (
        "WithinSubjectMatcher.match() debería lanzar ValueError con test_partition="
        "'loso_test-subject_08' (formato no reconocido), pero no lanzó nada."
    )
    assert "loso_test-subject_08" in error_message, (
        f"El mensaje de error debería mencionar el valor recibido ('loso_test-subject_08'), "
        f"mensaje real: {error_message}"
    )
    print(f"  ValueError lanzado correctamente: {error_message}")
    print("  OK\n")


def verify_unexpected_origin_partition_format_raises():
    print("--- 4. TrainedWeights con partition de ORIGEN de formato inesperado -> ValueError ---")
    matcher = WithinSubjectMatcher()
    # Acá el test_partition es válido, pero uno de los TrainedWeights viene
    # de un training con nombrado distinto (ej. mezclado por error, o un
    # training LOSO cuyos pesos terminaron en el mismo árbol de resultados).
    trained_weights = [
        _build_weights("subject_08/split_8_seed_399"),
        _build_weights("loso_test-subject_08/some_replicate"),  # formato de origen inesperado
    ]

    try:
        matcher.match(trained_weights, "subject_08")
        raised = False
    except ValueError as e:
        raised = True
        error_message = str(e)

    assert raised, (
        "WithinSubjectMatcher.match() debería lanzar ValueError al encontrar un TrainedWeights "
        "cuya partition de origen ('loso_test-subject_08/...') no tiene el formato subject_XX "
        "esperado, incluso si test_partition en sí es válido."
    )
    assert "loso_test-subject_08" in error_message, (
        f"El mensaje de error debería mencionar el segmento problemático, mensaje real: {error_message}"
    )
    print(f"  ValueError lanzado correctamente: {error_message}")
    print("  OK -- se valida tanto destino (test_partition) como origen (cada TrainedWeights)\n")


if __name__ == "__main__":
    verify_normal_case_matches_correctly()
    verify_no_match_returns_empty_list()
    verify_unexpected_test_partition_format_raises()
    verify_unexpected_origin_partition_format_raises()
    print("=== TODOS LOS CHECKS DE verify_09_weights_matcher_format.py PASARON ===")

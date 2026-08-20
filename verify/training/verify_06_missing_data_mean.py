"""
Script de verificación 6/6 -- MIBCI_SimpleDataset_MissingData, corrección
de la estrategia "mean" (constante por trial, no media por-instante).

Corré esto a mano (modo debug), no vía pytest.

Definición confirmada con el usuario: el relleno debe ser UNA ÚNICA
CONSTANTE ESCALAR POR TRIAL -- la media de TODOS los canales presentes,
colapsando canal Y tiempo juntos. Ese mismo valor (uno por trial) rellena
TODOS los canales faltantes en TODOS los instantes de ese trial -- no
varía en el tiempo.

El bug original calculaba mean(dim=1) sobre los canales presentes,
manteniendo la dimensión temporal -- daba una media que SÍ variaba en cada
instante (promedio entre canales presentes en ESE instante).

Qué verifica:
1. Para cada trial, TODOS los valores de relleno en los canales faltantes
   son exactamente el mismo número (constante en canal y tiempo) -- si el
   bug estuviera presente, estos valores variarían con el tiempo.
2. Ese valor constante coincide, con un cálculo hecho a mano de forma
   independiente, con la media de los canales presentes sobre TODAS las
   muestras (canal y tiempo juntos) de ESE trial.
3. Dos trials distintos tienen constantes de relleno DISTINTAS entre sí
   (si tienen datos distintos) -- confirma que es "por trial" y no una
   única constante para todo el dataset.

Nota de contrato encontrada al correr este script: la rama de expansión de
canales en MIBCI_SimpleDataset_MissingData.__init__ (cuando
X.shape[1] < len(originals_channels)) usa X.device -- asume que X ya es un
tensor de torch, a diferencia de MIBCI_SimpleDataset (la clase simple) que
acepta indistintamente np.ndarray o tensor. Acá, X_present se pasa como
torch.tensor explícito para respetar ese contrato real -- no es un bug a
corregir en este script, pero vale la pena saber que esta clase específica
NO acepta numpy crudo cuando hay canales faltantes que expandir.
"""
import numpy as np
import torch

from src.eeg_datasets.torch_datasets import MIBCI_SimpleDataset_MissingData


def verify_mean_is_constant_per_trial():
    print("--- 1 y 2. La constante de relleno es única por trial, coincide con el cálculo manual ---")

    originals_channels = ["C1", "C2", "C3", "C4"]
    channels_to_keep = ["C1", "C3"]  # faltan C2 y C4

    n_trials = 3
    n_time = 10

    rng = np.random.RandomState(0)
    # X de entrada: solo los canales presentes (C1, C3), shape (trials, 2, time).
    X_present = rng.uniform(low=1.0, high=100.0, size=(n_trials, len(channels_to_keep), n_time))

    ds = MIBCI_SimpleDataset_MissingData(
        torch.tensor(X_present.copy(), dtype=torch.float), y=np.array([0, 1, 0]), classification=True,
        originals_channels=originals_channels,
        channels_to_keep=channels_to_keep,
        missing_data_strategy="mean",
    )
    X_full = ds.X.numpy()  # (n_trials, 4, n_time) -- ya expandido y rellenado

    missing_indices = [originals_channels.index(ch) for ch in originals_channels if ch not in channels_to_keep]
    present_indices = [originals_channels.index(ch) for ch in channels_to_keep]

    constants_per_trial = []
    for trial_idx in range(n_trials):
        # --- Check 1: todos los valores de relleno del trial son EL MISMO número ---
        filled_values = X_full[trial_idx, missing_indices, :]  # (n_missing_channels, n_time)
        unique_values = np.unique(np.round(filled_values, decimals=6))
        assert len(unique_values) == 1, (
            f"Trial {trial_idx}: los valores de relleno NO son constantes -- hay {len(unique_values)} "
            f"valores distintos ({unique_values[:5]}...). Esperado: 1 solo valor (constante por trial, "
            f"sin variar en canal ni en tiempo)."
        )
        constant_value = unique_values[0]
        constants_per_trial.append(constant_value)

        # --- Check 2: ese valor coincide con el cálculo manual independiente ---
        # Media de TODOS los canales presentes y TODAS las muestras temporales
        # de ESE trial, colapsando ambas dimensiones juntas.
        present_data_this_trial = X_present[trial_idx]  # (n_present_channels, n_time), datos ORIGINALES
        expected_constant = present_data_this_trial.mean()  # escalar, colapsa canal Y tiempo

        assert abs(constant_value - expected_constant) < 1e-4, (
            f"Trial {trial_idx}: constante de relleno = {constant_value:.6f}, "
            f"esperada (media manual sobre canal+tiempo) = {expected_constant:.6f}"
        )
        print(f"  trial {trial_idx}: constante de relleno = {constant_value:.4f} "
              f"(esperado, cálculo manual = {expected_constant:.4f}) -- coincide")

        # --- Verificación adicional: los canales PRESENTES no fueron tocados ---
        present_values_after = X_full[trial_idx, present_indices, :]
        assert np.allclose(present_values_after, X_present[trial_idx]), (
            f"Trial {trial_idx}: los canales presentes fueron modificados -- no deberían tocarse."
        )

    print("  OK -- constante única por trial, coincide con el cálculo manual, canales presentes intactos\n")
    return constants_per_trial


def verify_constants_differ_across_trials(constants_per_trial):
    print("--- 3. Trials distintos (con datos distintos) tienen constantes DISTINTAS ---")
    unique_constants = np.unique(np.round(constants_per_trial, decimals=6))
    assert len(unique_constants) == len(constants_per_trial), (
        f"Se esperaban {len(constants_per_trial)} constantes distintas (una por trial, datos aleatorios "
        f"distintos por trial), se obtuvieron {len(unique_constants)} valores únicos: {unique_constants}. "
        f"¿La constante se está calculando sobre TODO el dataset en vez de por trial?"
    )
    print(f"  constantes por trial: {[f'{c:.4f}' for c in constants_per_trial]} -- todas distintas entre sí")
    print("  OK\n")


if __name__ == "__main__":
    constants = verify_mean_is_constant_per_trial()
    verify_constants_differ_across_trials(constants)
    print("=== TODOS LOS CHECKS DE verify_06_missing_data_mean.py PASARON ===")
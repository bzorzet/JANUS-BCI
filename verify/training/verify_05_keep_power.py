"""
Script de verificación 5/6 -- MultiFrequencyBandMaskingDataset, corrección
de keep_power (normalización por trial/canal, no global).

Corré esto a mano (modo debug), no vía pytest.

El bug original: keep_power calculaba UN escalar global (promedio de todo
el dataset) para reescalar la potencia post-filtrado, en vez de un factor
por trial y por canal. Este script construye trials con potencia MUY
distinta entre sí a propósito -- si el bug estuviera presente, el trial de
mayor potencia quedaría sub-compensado y el de menor potencia
sobre-compensado, visible al comparar potencia pre/post filtrado POR TRIAL
(no en promedio).

Qué verifica:
1. Con dos trials de potencia muy distinta (uno con amplitud x10 respecto
   al otro), la potencia POST-filtrado-y-reescalado de CADA trial se
   aproxima a su propia potencia ORIGINAL -- no al promedio de ambos.
2. Ídem por canal dentro de un mismo trial (dos canales de amplitud muy
   distinta).
3. Caso de referencia: si ambos trials tuvieran la MISMA potencia, el bug
   (normalización global) y el fix (normalización por trial) darían el
   mismo resultado -- este caso NO lo distingue, se incluye solo para
   confirmar que el caso "normal" sigue funcionando bien.
"""
import numpy as np

from src.eeg_datasets.torch_datasets import MultiFrequencyBandMaskingDataset


def _power_per_trial_channel(X):
    """X: (trials, channels, time). Devuelve potencia (trials, channels)."""
    return (X ** 2).sum(axis=-1)


def _simulate_eeg_like_signal(fs, n_samples, band_powers, rng, noise_level=0.3):
    """Simula una señal EEG-like: suma de sinusoides en bandas clásicas
    (delta/theta/alfa/beta/gamma) con potencias relativas dadas por
    `band_powers` (dict banda -> amplitud), más ruido rosa (1/f)
    característico del EEG real -- mucho más representativo que una sinusoide
    pura para este chequeo, porque la potencia queda distribuida en varias
    bandas a la vez, como en una señal real.

    band_powers: dict con claves de {"delta","theta","alpha","beta","gamma"}
    y valores = amplitud relativa de esa banda en este trial.
    """
    band_freqs = {"delta": 2.0, "theta": 6.0, "alpha": 10.0, "beta": 20.0, "gamma": 40.0}
    t = np.arange(n_samples) / fs

    signal = np.zeros(n_samples)
    for band, amplitude in band_powers.items():
        freq = band_freqs[band]
        # pequeña variación de fase aleatoria para que no sea perfectamente
        # periódica trial a trial, más parecido a EEG real.
        phase = rng.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Ruido 1/f (rosa): más energía en bajas frecuencias, característico
    # del EEG real -- generado en el dominio de la frecuencia y llevado a
    # tiempo con ifft.
    freqs = np.fft.rfftfreq(n_samples, d=1 / fs)
    freqs[0] = freqs[1]  # evitar división por cero en DC
    spectrum = (rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))) / freqs
    pink_noise = np.fft.irfft(spectrum, n=n_samples)
    pink_noise = pink_noise / pink_noise.std() * noise_level * np.std(signal)

    return signal + pink_noise


def verify_keep_power_per_trial_not_global():
    print("--- 1. keep_power reescala POR TRIAL, no con un promedio global (señal EEG-like) ---")

    fs = 128
    n_samples = 256
    rng = np.random.RandomState(42)

    # Trial 0: fuerte desincronización alfa/beta (típico de imaginería
    # motora activa) -- POCA potencia en la banda 8-30 Hz que se va a
    # eliminar, MÁS potencia en delta/theta/gamma.
    trial_motor_active = _simulate_eeg_like_signal(
        fs, n_samples,
        band_powers={"delta": 3.0, "theta": 2.0, "alpha": 0.5, "beta": 0.5, "gamma": 1.0},
        rng=rng,
    )
    # Trial 1: alfa/beta dominante (reposo, sin imaginería motora) -- MUCHA
    # potencia justo en la banda 8-30 Hz que se va a eliminar.
    trial_resting = _simulate_eeg_like_signal(
        fs, n_samples,
        band_powers={"delta": 1.0, "theta": 0.5, "alpha": 4.0, "beta": 3.0, "gamma": 0.3},
        rng=rng,
    )

    X = np.stack([trial_motor_active, trial_resting])[:, np.newaxis, :]  # (2 trials, 1 channel, n_samples)
    y = np.array([0, 1])

    filters = {"alpha_beta_kill": {"freqcut": [8.0, 30.0], "mask_type": "bandstop", "order": 4}}

    original_power = _power_per_trial_channel(X)  # (2, 1)

    ds = MultiFrequencyBandMaskingDataset(X.copy(), y, fs=fs, filters=filters, keep_power=True)
    X_processed = ds.X.numpy()
    processed_power = _power_per_trial_channel(X_processed)  # (2, 1)

    # Con el FIX (normalización por trial): cada trial debería recuperar
    # ~su propia potencia original, sea cual sea la potencia del otro trial
    # -- incluso aunque uno pierda MUCHA proporción de potencia (resting,
    # con alfa/beta dominante) y el otro pierda POCA (motor_active).
    ratio = processed_power / original_power  # debería ser ~1.0 para AMBOS trials

    print(f"  potencia original:  motor_active={original_power[0,0]:.2f}  resting={original_power[1,0]:.2f}")
    print(f"  potencia procesada: motor_active={processed_power[0,0]:.2f}  resting={processed_power[1,0]:.2f}")
    print(f"  ratio procesada/original: motor_active={ratio[0,0]:.4f}  resting={ratio[1,0]:.4f}")

    # Tolerancia generosa (15%) por el ruido 1/f y transitorios de filtro --
    # lo que NO debe pasar es que un trial esté cerca de 1.0 y el otro lejos
    # (firma del bug de normalización global, dado que ambos trials pierden
    # proporciones de potencia MUY distintas al filtrar).
    assert abs(ratio[0, 0] - 1.0) < 0.15, (
        f"Trial motor_active: ratio {ratio[0,0]:.4f} demasiado lejos de 1.0 -- "
        f"¿normalización global en vez de por trial?"
    )
    assert abs(ratio[1, 0] - 1.0) < 0.15, (
        f"Trial resting: ratio {ratio[1,0]:.4f} demasiado lejos de 1.0 -- "
        f"¿normalización global en vez de por trial?"
    )
    assert abs(ratio[0, 0] - ratio[1, 0]) < 0.1, (
        f"Los ratios de los dos trials difieren demasiado entre sí ({ratio[0,0]:.4f} vs {ratio[1,0]:.4f}) "
        f"-- señal de normalización global (bug), no por trial (fix esperado)."
    )
    print("  OK -- cada trial recupera su propia potencia, independiente de la potencia del otro trial\n")


def verify_keep_power_per_channel():
    print("--- 2. keep_power reescala POR CANAL dentro del mismo trial (señal EEG-like) ---")

    fs = 128
    n_samples = 256
    rng = np.random.RandomState(7)

    # Simula un canal motor (ej. C3, con desincronización alfa/beta -- poca
    # potencia en 8-30 Hz) y un canal más occipital/relajado (ej. O1, con
    # alfa dominante -- mucha potencia justo en la banda que se elimina),
    # dentro del MISMO trial. Composición espectral distinta entre canales,
    # no solo amplitud distinta -- mismo criterio que el test 1.
    ch_motor = _simulate_eeg_like_signal(
        fs, n_samples,
        band_powers={"delta": 2.5, "theta": 1.5, "alpha": 0.4, "beta": 0.4, "gamma": 0.8},
        rng=rng,
    )
    ch_occipital = _simulate_eeg_like_signal(
        fs, n_samples,
        band_powers={"delta": 0.8, "theta": 0.4, "alpha": 3.5, "beta": 2.5, "gamma": 0.2},
        rng=rng,
    )

    X = np.stack([ch_motor, ch_occipital])[np.newaxis, :, :]  # (1 trial, 2 channels, n_samples)
    y = np.array([0])

    filters = {"alpha_beta_kill": {"freqcut": [8.0, 30.0], "mask_type": "bandstop", "order": 4}}

    original_power = _power_per_trial_channel(X)  # (1, 2)
    ds = MultiFrequencyBandMaskingDataset(X.copy(), y, fs=fs, filters=filters, keep_power=True)
    processed_power = _power_per_trial_channel(ds.X.numpy())  # (1, 2)

    ratio = processed_power / original_power
    print(f"  potencia original:  motor={original_power[0,0]:.2f}  occipital={original_power[0,1]:.2f}")
    print(f"  potencia procesada: motor={processed_power[0,0]:.2f}  occipital={processed_power[0,1]:.2f}")
    print(f"  ratio procesada/original: motor={ratio[0,0]:.4f}  occipital={ratio[0,1]:.4f}")

    assert abs(ratio[0, 0] - 1.0) < 0.15, f"Canal motor: ratio {ratio[0,0]:.4f} lejos de 1.0"
    assert abs(ratio[0, 1] - 1.0) < 0.15, f"Canal occipital: ratio {ratio[0,1]:.4f} lejos de 1.0"
    assert abs(ratio[0, 0] - ratio[0, 1]) < 0.1, (
        f"Los ratios de los dos canales difieren demasiado ({ratio[0,0]:.4f} vs {ratio[0,1]:.4f}) "
        f"-- señal de normalización que no distingue por canal."
    )
    print("  OK -- cada canal recupera su propia potencia, independiente del otro canal\n")


def _apply_filters_raw(X, fs, filters):
    """Aplica el filtrado en cascada (SOS) SIN ningún reescalado de
    potencia -- mismo bloque 1-3 de _transform_data, extraído acá para
    poder comparar las dos estrategias de reescalado sobre el mismo
    resultado filtrado."""
    from scipy.signal import butter, sosfiltfilt

    processed = X.copy()
    for key, config in filters.items():
        sos = butter(config["order"], config["freqcut"], btype=config["mask_type"], fs=fs, output="sos")
        processed = sosfiltfilt(sos, processed, axis=-1)
    return np.ascontiguousarray(processed)


def _rescale_global_BUG(original, filtered):
    """Reimplementación INDEPENDIENTE del comportamiento VIEJO (con el bug):
    un solo escalar global, promedio de potencia sobre TODO el dataset
    (trials y canales juntos) -- exactamente la fórmula que tenía
    MultiFrequencyBandMaskingDataset ANTES de la corrección."""
    original_total_power = (original ** 2).sum(axis=-1).mean()
    processed_total_power = (filtered ** 2).sum(axis=-1).mean()
    if processed_total_power > 1e-10:
        scaling_factor = np.sqrt(original_total_power / processed_total_power)
        return filtered * scaling_factor
    return filtered


def _rescale_per_trial_channel_FIX(original, filtered):
    """Reimplementación INDEPENDIENTE del comportamiento NUEVO (fix): un
    factor por trial Y por canal -- misma fórmula que FrequencyMaskingDataset
    (la clase hermana que ya estaba correcta), ahora aplicada acá también."""
    power = (original ** 2).sum(axis=-1)              # (trials, channels)
    power_filtered = (filtered ** 2).sum(axis=-1)      # (trials, channels)
    k = np.sqrt(power / power_filtered)
    k = k[:, :, np.newaxis]
    return k * filtered


def verify_bug_vs_fix_side_by_side():
    print("--- 3. Comparación lado a lado: comportamiento VIEJO (bug) vs. NUEVO (fix), señal EEG-like ---")

    fs = 128
    n_samples = 256
    rng = np.random.RandomState(42)  # mismo seed que el test 1, mismo par de trials

    # Mismo par EEG-like que en el test 1: motor_active pierde POCA potencia
    # al filtrar 8-30 Hz (poca energía ahí), resting pierde MUCHA (alfa/beta
    # dominante) -- composición espectral genuinamente distinta, no solo
    # amplitud escalada.
    trial_motor_active = _simulate_eeg_like_signal(
        fs, n_samples,
        band_powers={"delta": 3.0, "theta": 2.0, "alpha": 0.5, "beta": 0.5, "gamma": 1.0},
        rng=rng,
    )
    trial_resting = _simulate_eeg_like_signal(
        fs, n_samples,
        band_powers={"delta": 1.0, "theta": 0.5, "alpha": 4.0, "beta": 3.0, "gamma": 0.3},
        rng=rng,
    )

    X = np.stack([trial_motor_active, trial_resting])[:, np.newaxis, :]  # (2 trials, 1 channel, n_samples)

    filters = {"alpha_beta_kill": {"freqcut": [8.0, 30.0], "mask_type": "bandstop", "order": 4}}
    filtered = _apply_filters_raw(X, fs, filters)  # mismo filtrado crudo para ambas comparaciones

    original_power = _power_per_trial_channel(X)
    filtered_power_raw = _power_per_trial_channel(filtered)  # ANTES de cualquier reescalado

    result_bug = _rescale_global_BUG(X, filtered)
    result_fix = _rescale_per_trial_channel_FIX(X, filtered)

    power_bug = _power_per_trial_channel(result_bug)
    power_fix = _power_per_trial_channel(result_fix)

    ratio_bug = power_bug / original_power
    ratio_fix = power_fix / original_power

    print(f"  potencia original:          motor_active={original_power[0,0]:.2f}   resting={original_power[1,0]:.2f}")
    print(f"  potencia post-filtro CRUDA: motor_active={filtered_power_raw[0,0]:.4f}   resting={filtered_power_raw[1,0]:.4f}  "
          f"(resting debería perder proporcionalmente MÁS potencia, por tener alfa/beta dominante)")
    print(f"  {'':25}{'BUG (global)':>18}{'FIX (por trial)':>20}")
    print(f"  ratio motor_active:      {ratio_bug[0,0]:>18.4f}{ratio_fix[0,0]:>20.4f}")
    print(f"  ratio resting:           {ratio_bug[1,0]:>18.4f}{ratio_fix[1,0]:>20.4f}")
    print(f"  diferencia entre trials: {abs(ratio_bug[0,0]-ratio_bug[1,0]):>18.4f}{abs(ratio_fix[0,0]-ratio_fix[1,0]):>20.4f}")

    # El bug debería mostrar una diferencia GRANDE entre los ratios de los
    # dos trials (uno sub-compensado, otro sobre-compensado, porque
    # perdieron proporciones de potencia muy distintas pero se reescalan
    # con el MISMO factor global); el fix debería mostrar ambos ratios
    # cerca de 1.0 y cerca entre sí (cada uno recupera SU propia potencia).
    diff_bug = abs(ratio_bug[0, 0] - ratio_bug[1, 0])
    diff_fix = abs(ratio_fix[0, 0] - ratio_fix[1, 0])

    assert diff_bug > diff_fix, (
        f"Se esperaba que el comportamiento BUG mostrara una diferencia mayor entre trials "
        f"({diff_bug:.4f}) que el FIX ({diff_fix:.4f}) -- si no, el caso de prueba elegido no "
        f"distingue bien entre ambas estrategias, revisar la composición espectral de los trials dummy."
    )
    assert diff_fix < 0.1, f"El FIX debería dar ratios muy parecidos entre trials, diferencia = {diff_fix:.4f}"

    print(f"\n  el FIX reduce la diferencia entre trials de {diff_bug:.4f} (bug) a {diff_fix:.4f} (fix)")
    print("  esto confirma visualmente que la clase real (MultiFrequencyBandMaskingDataset) debe")
    print("  comportarse como la columna FIX, no como la columna BUG.")
    print("  OK\n")


if __name__ == "__main__":
    verify_keep_power_per_trial_not_global()
    verify_keep_power_per_channel()
    verify_bug_vs_fix_side_by_side()
    print("=== TODOS LOS CHECKS DE verify_05_keep_power.py PASARON ===")
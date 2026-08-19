"""
Script de verificación adicional -- keep_power sobre DATOS REALES
preprocesados (no dummy data). Complementa a verify_05_keep_power.py: ahí
se usó señal EEG-like sintética para aislar el mecanismo del bug con
control total sobre la composición espectral; acá se confirma que la
diferencia bug/fix es visible y tiene sentido en tus datos preprocesados
reales.

Corré esto a mano (modo debug), no vía pytest. No hace asserts duros de
paso/fallo -- es exploratorio, para inspección visual.

QUÉ HACE:
1. Instancia PreprocessedDataset UNA vez con los parámetros seteados abajo
   (EDITAR antes de correr).
2. Para CADA sujeto de SUBJECTS por separado (flatten_subject_data, no
   flatten_pool_data -- coherente con producción, donde
   MultiFrequencyBandMaskingDataset se instancia una vez POR SUJETO, así
   que el escalar global del bug se calculaba sobre el pool de ESE sujeto
   únicamente):
   a. Carga sus trials, selecciona los 3 canales configurados.
   b. Para cada condición de ablación en ABLATION_CONFIGS, calcula
      potencia por trial bajo tres variantes:
        - "original": señal sin filtrar.
        - "bug": filtrada + reescalada con la fórmula VIEJA (normalización
          global, un escalar para todo el POOL DE ESE SUJETO).
        - "fix": filtrada + reescalada con la fórmula NUEVA (normalización
          por trial y canal) -- la que ya está en el código real.
   c. Genera UNA FIGURA COMPLETA por sujeto (boxplot con seaborn, eje X =
      canal, hue = {original, bug, fix}, un subplot por ablación).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.eeg_datasets import PreprocessedDataset  # ajustar el import si el path real difiere

# ============================================================
# EDITAR ANTES DE CORRER
# ============================================================
PREPROCESSING_NAME = "simple-CAR-128Hz-preproc"
DB_NAME = "Cho2017"
SESSION = "session_1"
SUBJECTS = [3, 14, 26, 28, 35, 37, 39, 43, 44, 47, 49, 50]    # lista de subject_id a incluir, o None para todos
CHANNELS_TO_PLOT = ["C3", "Cz", "C4"]  # deben existir en el dataset preprocesado
FS = 128.0                     # debe coincidir con el sfreq real del dataset preprocesado
DATA_TO_LOAD = ["motor_imagery"]  # debe coincidir con lo que se guardó en el preprocesamiento
# Condiciones de ablación a comparar -- cada una es un subplot. Tomadas
# directamente de los configs JSON reales (torch_dataset.params.filters).
ABLATION_CONFIGS = {
    # Config real de _dtg.json: un solo bandstop, elimina alpha+beta y
    # deja el resto del espectro intacto ("ablation" -- se saca la banda).
    "alpha_beta_ablation (dtg)": {
        "alpha_beta_kill": {"freqcut": [8.0, 30.0], "mask_type": "bandstop", "order": 4},
    },
    # Config real de aislamiento: 3 filtros en cascada -- bandpass que
    # DEJA pasar alpha+beta, más dos bandstop que eliminan todo lo demás
    # (delta+theta por un lado, gamma por el otro). El resultado final
    # debería tener casi toda su potencia concentrada en 8-30 Hz.
    "alpha_beta_isolation": {
        "alpha_beta_pass": {"freqcut": [8.0, 30.0], "mask_type": "bandpass", "order": 4},
        "delta_theta_kill": {"freqcut": [1.0, 7.5], "mask_type": "bandstop", "order": 4},
        "gamma_kill": {"freqcut": [30.5, 45.0], "mask_type": "bandstop", "order": 4},
    },
    # Agregar acá más condiciones si hace falta comparar otra ablación,
    # con el mismo formato que "filters" en el JSON real.
}
# ============================================================


def _power_per_trial_channel(X):
    """X: (trials, channels, time). Devuelve potencia (trials, channels)."""
    return (X ** 2).sum(axis=-1)


def _apply_filters_raw(X, fs, filters):
    """Filtrado en cascada (SOS), sin ningún reescalado -- idéntico al
    bloque 1-3 de MultiFrequencyBandMaskingDataset._transform_data."""
    from scipy.signal import butter, sosfiltfilt

    processed = X.copy()
    for key, config in filters.items():
        sos = butter(config["order"], config["freqcut"], btype=config["mask_type"], fs=fs, output="sos")
        processed = sosfiltfilt(sos, processed, axis=-1)
    return np.ascontiguousarray(processed)


def _rescale_global_BUG(original, filtered):
    """Fórmula VIEJA: un escalar global (promedio de potencia sobre TODO
    el dataset -- trials y canales juntos)."""
    original_total_power = (original ** 2).sum(axis=-1).mean()
    processed_total_power = (filtered ** 2).sum(axis=-1).mean()
    if processed_total_power > 1e-10:
        scaling_factor = np.sqrt(original_total_power / processed_total_power)
        return filtered * scaling_factor
    return filtered


def _rescale_per_trial_channel_FIX(original, filtered):
    """Fórmula NUEVA: un factor por trial Y por canal -- la que ya está
    en el código real."""
    power = (original ** 2).sum(axis=-1)
    power_filtered = (filtered ** 2).sum(axis=-1)
    k = np.sqrt(power / power_filtered)
    k = k[:, :, np.newaxis]
    return k * filtered


def load_subject_data(dataset, subject_id, channel_indices):
    """Carga UN sujeto por vez (flatten_subject_data, no flatten_pool_data)
    -- coherente con producción: MultiFrequencyBandMaskingDataset se
    instancia una vez POR SUJETO, así que el escalar global del bug
    (y el factor por trial/canal del fix) deben calcularse sobre el pool
    de ESE sujeto únicamente, no sobre todos los sujetos mezclados."""
    X, y, metadata = dataset.flatten_subject_data(subject_id, session=SESSION)
    X_selected = X[:, channel_indices, :]
    print(f"  sujeto {subject_id}: X_selected.shape={X_selected.shape}")
    return X_selected


def setup_dataset():
    print(f"Instanciando PreprocessedDataset: db={DB_NAME}, preprocessing={PREPROCESSING_NAME}")
    dataset = PreprocessedDataset(
        preprocessing_name=PREPROCESSING_NAME,
        db_name=DB_NAME,
        sessions=[SESSION],
        subjects=SUBJECTS,
        data_to_load=["motor_imagery"],
        channels=None,  # traemos todos los canales acá, filtramos a los 3 elegidos después
        classes_to_return=None,
    )

    if not all(ch in dataset.ch_names for ch in CHANNELS_TO_PLOT):
        missing = [ch for ch in CHANNELS_TO_PLOT if ch not in dataset.ch_names]
        raise ValueError(
            f"Canales {missing} no existen en el dataset preprocesado. "
            f"Canales disponibles: {dataset.ch_names}"
        )
    channel_indices = [dataset.ch_names.index(ch) for ch in CHANNELS_TO_PLOT]
    return dataset, channel_indices


def build_ratio_dataframe(X_selected, fs, ablation_name, filters):
    """Devuelve un DataFrame largo con el RATIO power/power_original (no
    la potencia absoluta): columnas [channel, condition, ratio], una fila
    por (trial, channel, condition). Se grafica el ratio en vez de la
    potencia absoluta porque la dispersión real de la potencia ORIGINAL
    (sin filtrar) es muy chica comparada con la dispersión post-filtro
    (bug/fix) -- graficar las tres en la misma escala de potencia absoluta
    aplasta visualmente la variación real de 'original' (se ve como una
    línea plana). El ratio no tiene ese problema: por definición,
    ratio_original == 1.0 siempre (no se grafica, es trivial), y
    ratio_bug/ratio_fix se leen directamente como "cuánto se aleja de la
    potencia real de ESE trial/canal" -- 1.0 es el valor ideal."""
    filtered_raw = _apply_filters_raw(X_selected, fs, filters)
    X_bug = _rescale_global_BUG(X_selected, filtered_raw)
    X_fix = _rescale_per_trial_channel_FIX(X_selected, filtered_raw)

    power_original = _power_per_trial_channel(X_selected)  # (trials, channels)
    power_bug = _power_per_trial_channel(X_bug)
    power_fix = _power_per_trial_channel(X_fix)

    ratio_bug = power_bug / power_original
    ratio_fix = power_fix / power_original

    rows = []
    n_trials, n_channels = power_original.shape
    for condition_name, ratio_matrix in [("bug", ratio_bug), ("fix", ratio_fix)]:
        for trial_idx in range(n_trials):
            for ch_idx in range(n_channels):
                rows.append({
                    "ablation": ablation_name,
                    "channel": CHANNELS_TO_PLOT[ch_idx],
                    "condition": condition_name,
                    "ratio": ratio_matrix[trial_idx, ch_idx],
                    "trial": trial_idx,
                })
    return pd.DataFrame(rows)


def plot_ratio_comparison(df, subject_id):
    ablations = df["ablation"].unique()
    n_ablations = len(ablations)

    fig, axes = plt.subplots(1, n_ablations, figsize=(6 * n_ablations, 5), squeeze=False)
    axes = axes[0]

    for ax, ablation_name in zip(axes, ablations):
        subset = df[df["ablation"] == ablation_name]
        sns.boxplot(
            data=subset, x="channel", y="ratio", hue="condition",
            order=CHANNELS_TO_PLOT, hue_order=["bug", "fix"],
            ax=ax, showfliers=False,
        )
        sns.stripplot(
            data=subset, x="channel", y="ratio", hue="condition",
            order=CHANNELS_TO_PLOT, hue_order=["bug", "fix"],
            dodge=True, alpha=0.4, size=3, ax=ax, legend=False,
        )
        # Línea de referencia: ratio=1.0 es el valor ideal (potencia
        # post-filtro-y-reescalado == potencia original de ESE trial/canal).
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, zorder=0)
        ax.set_title(ablation_name)
        ax.set_ylabel("Ratio potencia (post-reescalado / original), por trial")
        ax.set_xlabel("Canal")

    fig.suptitle(f"Ratio de potencia por trial y canal: bug (normalización global) vs. fix (por trial/canal)\n"
                 f"Línea punteada = 1.0 (potencia original preservada). Dataset: {DB_NAME} | Sujeto: {subject_id}",
                 fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    dataset, channel_indices = setup_dataset()
    subjects_to_use = SUBJECTS if SUBJECTS is not None else dataset.subject_list

    for subject_id in subjects_to_use:
        print(f"\n{'='*60}\nSujeto {subject_id}\n{'='*60}")
        X_selected = load_subject_data(dataset, subject_id, channel_indices)

        all_dfs = []
        for ablation_name, filters in ABLATION_CONFIGS.items():
            print(f"  procesando ablación: {ablation_name}")
            df_ablation = build_ratio_dataframe(X_selected, FS, ablation_name, filters)
            all_dfs.append(df_ablation)

            # Resumen numérico rápido en consola, además del gráfico -- para
            # poder leer valores concretos sin depender solo del boxplot.
            # El "std" acá es la métrica clave: bug debería tener std mucho
            # mayor que fix (más dispersión = reescalado menos fiel al
            # trial/canal real), y la media de fix debería estar mucho más
            # cerca de 1.0 que la de bug.
            summary = df_ablation.groupby(["channel", "condition"])["ratio"].agg(["mean", "median", "std"])
            print(summary)

        df_full = pd.concat(all_dfs, ignore_index=True)
        fig = plot_ratio_comparison(df_full, subject_id)
        plt.show()

    print("\nListo -- inspeccioná los gráficos, uno por sujeto. En cada uno, 'bug' debería mostrar")
    print("mayor dispersión y alejarse más de la línea ratio=1.0 que 'fix', que debería mantenerse")
    print("cerca de 1.0 (con dispersión chica) en los tres canales.")
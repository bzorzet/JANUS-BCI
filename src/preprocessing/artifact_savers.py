"""
Registro de savers de artifacts pesados producidos por un stage (ver
preprocessing/DESIGN.md sección 4 punto 3). Cada saver recibe el valor
del artifact más el resto de artifacts/detail_tables del mismo stage,
para poder cruzar información — ej. el saver de "ica_obj" necesita
`raw_before_ica` (otro artifact del mismo stage) y `component_labels`
(un detail_table del mismo stage) para generar la auditoría visual.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List


def save_ica_artifacts(
    value: Any,
    *,
    artifacts: Dict[str, Any],
    detail_tables: Dict[str, List[dict]],
    output_dir: Path,
    prefix: str,
) -> None:
    """Puerto de _generate_ica_visuals/_plot_ica_scores (repo_viejo/src/ica/ica_database_preprocessor.py)."""
    import matplotlib.pyplot as plt  # deferred: heavy/optional dep

    ica = value
    output_dir.mkdir(parents=True, exist_ok=True)
    ica.save(output_dir / f"{prefix}_ica.fif", overwrite=True, verbose=False)

    raw_before = artifacts.get("raw_before_ica")
    if raw_before is None:
        return

    component_labels = detail_tables.get("component_labels", [])
    labels = [row["label"] for row in component_labels]
    probabilities = [row["probability"] for row in component_labels]
    excluded = [row["component"] for row in component_labels if row["excluded"]]

    fig_topo = ica.plot_components(show=False, title=f"ICA Components - {prefix}")
    fig_topo.savefig(output_dir / f"{prefix}_all_topos.png")

    if excluded:
        fig_props = ica.plot_properties(raw_before, picks=excluded, show=False)
        for comp_id, fig in zip(excluded, fig_props):
            fig.savefig(output_dir / f"{prefix}_prop_comp_{comp_id}.png")

    fig_overlay = ica.plot_overlay(raw_before, exclude=excluded, show=False, title=f"Overlay - {prefix}")
    fig_overlay.savefig(output_dir / f"{prefix}_signal_overlay.png")

    _plot_ica_scores(labels, probabilities, excluded, output_dir, prefix)
    plt.close("all")


def _plot_ica_scores(labels, probabilities, excluded, output_dir: Path, prefix: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    comps = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["red" if i in excluded else "green" for i in comps]
    ax.bar(comps, probabilities, color=colors, alpha=0.7)
    ax.set_xticks(comps)
    ax.set_xticklabels([f"IC{i}\n{labels[i]}" for i in comps], fontsize=8)
    ax.set_ylabel("Confidence (Probability)")
    ax.set_title(f"ICA Label Scores - {prefix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_label_scores.png")
    plt.close(fig)


# "raw_before_ica" no tiene saver propio a propósito: lo consume
# save_ica_artifacts como insumo, no se guarda por separado. El
# orquestador loguea un aviso informativo y sigue cuando una key no
# tiene saver — comportamiento esperado, no un error.
ARTIFACT_SAVERS: Dict[str, Callable] = {
    "ica_obj": save_ica_artifacts,
}

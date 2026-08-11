"""
Progreso interactivo para EEGDatabasePreprocessor (DESIGN.md, cambio
acotado sobre el orquestador existente). `ProgressReporter` es el
contrato — no-op por default para no romper cron ni logs de servidor.
`RichProgressReporter` es la única implementación real, para debug
interactivo en terminal.
"""
from typing import Any, Dict, Optional


class ProgressReporter:
    """No-op por default. EEGDatabasePreprocessor siempre tiene un
    reporter (nunca None) — este es el que usa cuando no se pidió uno
    explícito."""

    def on_start(self, total: int) -> None:
        pass

    def on_run_start(self, partition: str) -> None:
        pass

    def on_run_result(self, partition: str, status: str) -> None:
        pass

    def on_finish(self, summary: dict) -> None:
        pass


# (color, ícono) por estado. "pendiente" queda definido acá aunque hoy no
# se alcanza: el orquestador solo sabe que un run existe en el momento en
# que lo carga, así que una fila nace directamente en "corriendo" — no hay
# forma de anunciarla antes sin cargar el subject dos veces.
_STATUS_STYLE = {
    "pendiente": ("dim", "…"),
    "corriendo": ("yellow", "⏳"),
    "success": ("green", "✓"),
    "failed": ("red", "✗"),
    "omitido": ("cyan", "⏭"),
}


class RichProgressReporter(ProgressReporter):
    """Tabla en vivo (una fila por subject×context×run) + barra de
    progreso global, vía rich.live.Live. La barra crece de forma dinámica
    con cada on_run_start (nueva fila conocida) y avanza con cada
    on_run_result (fila resuelta) — llega a 100% en on_finish sin
    necesidad de conocer el total de runs de antemano."""

    def __init__(self, console: Optional[Any] = None) -> None:
        from rich.console import Console

        self.console = console or Console()
        self._live = None
        self._progress = None
        self._task_id = None
        self._rows: Dict[str, Dict[str, str]] = {}
        self._runs_known = 0
        self._last_refresh = 0.0

    def _build_table(self):
        from rich.table import Table

        table = Table(expand=True)
        table.add_column("Subject")
        table.add_column("Context")
        table.add_column("Run")
        table.add_column("Estado")
        for partition, row in self._rows.items():
            color, icon = _STATUS_STYLE.get(row["status"], ("white", "?"))
            table.add_row(
                row["subject"], row["context"], row["run_id"],
                f"[{color}]{icon} {row['status']}[/{color}]",
            )
        return table

    def _refresh(self, force: bool = False) -> None:
        import time

        from rich.console import Group

        # Reconstruir la tabla es O(filas conocidas) — en un resume grande
        # (miles de "omitido" seguidos, sin trabajo real entre medio) llamar
        # esto en cada evento sería O(n²). Se throttlea por tiempo (no por
        # cantidad de eventos) para no demorar el feedback de un run real,
        # que sí tarda; el estado interno (self._rows) igual queda al día
        # en cada evento, solo se pospone el redibujado.
        now = time.monotonic()
        if not force and (now - self._last_refresh) < 0.25:
            return
        self._last_refresh = now
        if self._live is not None:
            self._live.update(Group(self._progress, self._build_table()))

    @staticmethod
    def _parse_partition(partition: str):
        parts = partition.split("/")
        while len(parts) < 3:
            parts.append("unknown")
        return parts[0], parts[1], parts[2]

    def on_start(self, total: int) -> None:
        from rich.live import Live
        from rich.progress import Progress

        self._progress = Progress(
            *Progress.get_default_columns(),
            "•",
            f"{total} combinaciones subject×session",
        )
        self._task_id = self._progress.add_task("Preprocesando", total=0)
        self._live = Live(self._progress, console=self.console, refresh_per_second=4)
        self._live.start()

    def on_run_start(self, partition: str) -> None:
        subject, context, run_id = self._parse_partition(partition)
        self._rows[partition] = {
            "subject": subject, "context": context, "run_id": run_id,
            "status": "corriendo",
        }
        self._runs_known += 1
        self._progress.update(self._task_id, total=self._runs_known)
        self._refresh()

    def on_run_result(self, partition: str, status: str) -> None:
        if partition in self._rows:
            self._rows[partition]["status"] = status
        self._progress.advance(self._task_id)
        self._refresh()

    def on_finish(self, summary: dict) -> None:
        from rich.panel import Panel

        if self._live is not None:
            self._live.stop()
        lines = [f"{key}: {value}" for key, value in summary.items()]
        self.console.print(Panel("\n".join(lines), title="Resumen del barrido"))

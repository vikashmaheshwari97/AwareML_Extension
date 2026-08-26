from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional
import platform
import time
import psutil


@dataclass
class SustainabilityRecord:
    status: str
    started_utc: str
    ended_utc: str
    duration_sec: float
    energy_kwh: Optional[float]
    co2_kg: Optional[float]
    measurement_backend: str
    country_iso: str
    cpu: str
    logical_cpus: int
    ram_gb: float
    python: str
    codecarbon_version: Optional[str]
    gpu: Optional[str]
    notes: list[str]

    def to_dict(self):
        return asdict(self)


class SustainabilitySession:
    """CodeCarbon measurement with explicit missing-state semantics.

    A user request to measure sustainability is sufficient to start CodeCarbon; there is no
    second hidden environment-variable gate. If CodeCarbon is unavailable or cannot measure,
    the record stays ``None``/``not_measured`` rather than substituting zero.
    """

    def __init__(self, enabled: bool = False, country_iso: str = "EST", project_name: str = "AwareML-Extension"):
        self.enabled = bool(enabled)
        self.country_iso = country_iso
        self.project_name = project_name
        self.tracker = None
        self.start_t = None
        self.start_utc = None
        self.notes: list[str] = []
        self.codecarbon_version = None
        self.gpu = None

    def start(self):
        self.start_t = time.perf_counter()
        self.start_utc = datetime.now(timezone.utc).isoformat()
        if not self.enabled:
            self.notes.append("Measurement was not requested for this run.")
            return self
        try:
            import codecarbon
            from codecarbon import OfflineEmissionsTracker
            self.codecarbon_version = getattr(codecarbon, "__version__", None)
            self.tracker = OfflineEmissionsTracker(
                project_name=self.project_name,
                country_iso_code=self.country_iso,
                save_to_file=False,
                log_level="error",
                allow_multiple_runs=True,
            )
            self.tracker.start()
        except Exception as e:
            self.tracker = None
            self.notes.append(f"CodeCarbon could not start: {type(e).__name__}: {e}")
        return self

    @staticmethod
    def _float_or_none(value):
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            try:
                return float(getattr(value, "value"))
            except Exception:
                return None

    def stop(self) -> SustainabilityRecord:
        duration = max(0.0, time.perf_counter() - (self.start_t or time.perf_counter()))
        energy = None
        co2 = None
        status = "not_measured"
        backend = "none"
        if self.tracker is not None:
            backend = "CodeCarbon OfflineEmissionsTracker"
            try:
                emissions = self.tracker.stop()
                co2 = self._float_or_none(emissions)
                final = getattr(self.tracker, "final_emissions_data", None)
                energy = self._float_or_none(getattr(final, "energy_consumed", None)) if final is not None else None
                gpu_energy = self._float_or_none(getattr(final, "gpu_energy", None)) if final is not None else None
                cpu_energy = self._float_or_none(getattr(final, "cpu_energy", None)) if final is not None else None
                if gpu_energy is not None:
                    self.notes.append(f"GPU energy component: {gpu_energy:.8g} kWh")
                if cpu_energy is not None:
                    self.notes.append(f"CPU energy component: {cpu_energy:.8g} kWh")
                status = "measured" if (energy is not None or co2 is not None) else "measurement_incomplete"
                if status != "measured":
                    self.notes.append("CodeCarbon completed but did not expose energy or CO2 values.")
            except Exception as e:
                self.notes.append(f"CodeCarbon failed to stop cleanly: {type(e).__name__}: {e}")
                status = "measurement_failed"

        # Best-effort GPU inventory, independent of CodeCarbon measurement success.
        try:
            import pynvml
            pynvml.nvmlInit()
            names = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                n = pynvml.nvmlDeviceGetName(h)
                names.append(n.decode() if isinstance(n, bytes) else str(n))
            self.gpu = ", ".join(names) if names else None
        except Exception:
            self.gpu = None

        return SustainabilityRecord(
            status=status,
            started_utc=self.start_utc or datetime.now(timezone.utc).isoformat(),
            ended_utc=datetime.now(timezone.utc).isoformat(),
            duration_sec=duration,
            energy_kwh=energy,
            co2_kg=co2,
            measurement_backend=backend,
            country_iso=self.country_iso,
            cpu=platform.processor() or platform.machine(),
            logical_cpus=psutil.cpu_count(logical=True) or 0,
            ram_gb=round(psutil.virtual_memory().total / (1024 ** 3), 2),
            python=platform.python_version(),
            codecarbon_version=self.codecarbon_version,
            gpu=self.gpu,
            notes=self.notes,
        )

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    region: Optional[str]
    carbon_intensity_g_per_kwh: Optional[float]
    carbon_intensity_source: Optional[str]
    cpu: str
    physical_cpus: int
    logical_cpus: int
    ram_gb: float
    gpu: Optional[str]
    python: str
    codecarbon_version: Optional[str]
    warmup_sec: float
    warmup_samples: int
    repetition_id: int
    repetitions_planned: int
    measurement_failure_reason: Optional[str]
    notes: list[str]

    def to_dict(self):
        return asdict(self)


class SustainabilitySession:
    """CodeCarbon measurement with explicit Phase-14 protocol semantics."""

    def __init__(
        self,
        enabled: bool = False,
        country_iso: str = "EST",
        project_name: str = "AwareML-Extension",
        *,
        region: Optional[str] = None,
        warmup_sec: float = 0.0,
        warmup_samples: int = 0,
        repetition_id: int = 1,
        repetitions_planned: int = 1,
    ):
        self.enabled = bool(enabled)
        self.country_iso = str(country_iso)
        self.project_name = str(project_name)
        self.region = region
        self.warmup_sec = max(0.0, float(warmup_sec))
        self.warmup_samples = max(0, int(warmup_samples))
        self.repetition_id = max(1, int(repetition_id))
        self.repetitions_planned = max(
            self.repetition_id, int(repetitions_planned)
        )
        self.tracker = None
        self.start_t = None
        self.start_utc = None
        self.notes = []
        self.codecarbon_version = None
        self.gpu = None
        self.measurement_failure_reason = None

    def start(self):
        if self.enabled and self.warmup_sec > 0:
            time.sleep(self.warmup_sec)
            self.notes.append(
                "Stabilization delay of {:.3f} s completed before measurement."
                .format(self.warmup_sec)
            )

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
                allow_multiple_runs=False,
            )
            self.tracker.start()
        except Exception as exc:
            self.tracker = None
            self.measurement_failure_reason = "{}: {}".format(
                type(exc).__name__, exc
            )
            self.notes.append(
                "CodeCarbon could not start: {}".format(
                    self.measurement_failure_reason
                )
            )
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

    @staticmethod
    def _derive_carbon_intensity(energy_kwh, co2_kg):
        if energy_kwh is None or co2_kg is None:
            return None
        try:
            energy = float(energy_kwh)
            co2 = float(co2_kg)
        except Exception:
            return None
        if energy <= 0 or co2 < 0:
            return None
        return float((co2 * 1000.0) / energy)

    @staticmethod
    def _gpu_inventory():
        try:
            import pynvml
            pynvml.nvmlInit()
            names = []
            for idx in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(handle)
                names.append(
                    name.decode() if isinstance(name, bytes) else str(name)
                )
            return ", ".join(names) if names else None
        except Exception:
            return None

    def stop(self):
        now = time.perf_counter()
        duration = max(
            0.0,
            now - (self.start_t if self.start_t is not None else now),
        )
        energy = None
        co2 = None
        carbon_intensity = None
        carbon_intensity_source = None
        status = "not_measured"
        backend = "none"
        detected_region = self.region

        if self.enabled and self.tracker is None:
            status = "measurement_failed"

        if self.tracker is not None:
            backend = "CodeCarbon OfflineEmissionsTracker"
            try:
                emissions = self.tracker.stop()
                co2 = self._float_or_none(emissions)
                final = getattr(self.tracker, "final_emissions_data", None)

                energy = (
                    self._float_or_none(
                        getattr(final, "energy_consumed", None)
                    )
                    if final is not None else None
                )
                gpu_energy = (
                    self._float_or_none(getattr(final, "gpu_energy", None))
                    if final is not None else None
                )
                cpu_energy = (
                    self._float_or_none(getattr(final, "cpu_energy", None))
                    if final is not None else None
                )

                if final is not None and detected_region is None:
                    for attr in ("region", "country_name"):
                        value = getattr(final, attr, None)
                        if value:
                            detected_region = str(value)
                            break

                # Unit-safe carbon intensity. Do not reuse emissions_rate.
                carbon_intensity = self._derive_carbon_intensity(
                    energy, co2
                )
                if carbon_intensity is not None:
                    carbon_intensity_source = (
                        "derived_from_measured_co2_and_energy"
                    )

                if gpu_energy is not None:
                    self.notes.append(
                        "GPU energy component: {:.8g} kWh".format(gpu_energy)
                    )
                if cpu_energy is not None:
                    self.notes.append(
                        "CPU energy component: {:.8g} kWh".format(cpu_energy)
                    )

                status = (
                    "measured"
                    if energy is not None and co2 is not None
                    else "measurement_incomplete"
                )
                if status != "measured":
                    self.notes.append(
                        "CodeCarbon completed but did not expose both energy and CO2."
                    )
            except Exception as exc:
                self.measurement_failure_reason = "{}: {}".format(
                    type(exc).__name__, exc
                )
                self.notes.append(
                    "CodeCarbon failed to stop cleanly: {}".format(
                        self.measurement_failure_reason
                    )
                )
                status = "measurement_failed"

        self.gpu = self._gpu_inventory()

        return SustainabilityRecord(
            status=status,
            started_utc=self.start_utc
            or datetime.now(timezone.utc).isoformat(),
            ended_utc=datetime.now(timezone.utc).isoformat(),
            duration_sec=float(duration),
            energy_kwh=energy,
            co2_kg=co2,
            measurement_backend=backend,
            country_iso=self.country_iso,
            region=detected_region,
            carbon_intensity_g_per_kwh=carbon_intensity,
            carbon_intensity_source=carbon_intensity_source,
            cpu=platform.processor() or platform.machine() or "unknown",
            physical_cpus=int(psutil.cpu_count(logical=False) or 0),
            logical_cpus=int(psutil.cpu_count(logical=True) or 0),
            ram_gb=float(psutil.virtual_memory().total / (1024 ** 3)),
            gpu=self.gpu,
            python=platform.python_version(),
            codecarbon_version=self.codecarbon_version,
            warmup_sec=self.warmup_sec,
            warmup_samples=self.warmup_samples,
            repetition_id=self.repetition_id,
            repetitions_planned=self.repetitions_planned,
            measurement_failure_reason=self.measurement_failure_reason,
            notes=self.notes,
        )

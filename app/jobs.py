"""Registro de trabajos y pipeline de render ejecutado en segundo plano."""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .exceptions import GPXRenderError
from .gpx_processor import ProcessedTrack
from .renderer_orchestrator import RendererOrchestrator
from .schemas import JobStatus, RenderOptions
from .video_library import VideoLibrary
from .video_compiler import StreamingVideoCompiler

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    """Estado interno de un trabajo de render."""

    job_id: str
    options: RenderOptions
    track: ProcessedTrack
    title: str = "ruta"
    owner: str | None = None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    error: str | None = None
    video_path: Path | None = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)


class JobStore:
    """Almacén en memoria, seguro entre hilos.

    Válido para un único proceso. Con varios workers de Uvicorn o varias
    réplicas debe sustituirse por Redis/BD (la interfaz es deliberadamente mínima).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()


class RenderPipeline:
    """Ejecuta render + compilación de un trabajo dentro de un directorio temporal propio."""

    def __init__(self, store: JobStore, settings: Settings, library: VideoLibrary) -> None:
        self._store = store
        self._library = library
        self._settings = settings
        # Limita los renders simultáneos: son intensivos en CPU y disco.
        self._slots = threading.BoundedSemaphore(settings.max_concurrent_renders)

    def run(self, job_id: str) -> None:
        """Punto de entrada síncrono (FastAPI lo ejecuta en su threadpool).

        Nunca propaga excepciones: cualquier fallo queda registrado en el trabajo.
        """
        job = self._store.get(job_id)
        if job is None:
            return
        settings = self._settings
        workdir = Path(tempfile.mkdtemp(prefix=f"gpxrender_{job_id}_"))
        try:
            with self._slots:
                self._store.update(job_id, status=JobStatus.RENDERING)
                orchestrator = RendererOrchestrator(settings=settings)
                payload = orchestrator.build_payload(job_id, job.track, job.options)

                compiler = StreamingVideoCompiler(
                    settings.output_dir / f"{job_id}.mp4",
                    width=payload.width,
                    height=payload.height,
                    fps=payload.fps,
                    crf=settings.ffmpeg_crf,
                    preset=settings.ffmpeg_preset,
                    ffmpeg_binary=settings.ffmpeg_binary,
                    timeout_s=settings.ffmpeg_timeout_s,
                )
                frames_dir = workdir / "frames"
                with compiler:

                    def on_progress(done: int, total: int) -> None:
                        # Cada lote terminado se codifica y se borra: el disco temporal no crece.
                        compiler.feed(frames_dir, done)
                        self._store.update(job_id, progress=round(0.97 * done / total, 3))

                    orchestrator.render_frames(payload, workdir, on_progress, verify=False)
                    compiler.feed(frames_dir, payload.frame_count)  # motores que no informan del progreso
                    self._store.update(job_id, status=JobStatus.COMPILING, progress=0.97)
                    video = compiler.finish(payload.frame_count)
                self._library.register(
                    job_id,
                    video,
                    owner=job.owner,
                    title=job.title,
                    options=job.options,
                    metrics=job.track.metrics,
                    duration_s=payload.duration_s,
                    width=payload.width,
                    height=payload.height,
                )
            self._store.update(job_id, status=JobStatus.COMPLETED, progress=1.0, video_path=video)
            logger.info("Job %s completado: %s", job_id, video)
        except GPXRenderError as exc:
            logger.error("Job %s falló: %s | %s", job_id, exc.message, exc.detail)
            self._store.update(job_id, status=JobStatus.FAILED, error=exc.message)
        except Exception:  # noqa: BLE001
            logger.exception("Job %s: error inesperado", job_id)
            self._store.update(job_id, status=JobStatus.FAILED, error="Error interno durante el render.")
        finally:
            # Payload, fotogramas residuales y cualquier temporal desaparecen pase lo que pase.
            shutil.rmtree(workdir, ignore_errors=True)

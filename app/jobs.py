"""Registro de trabajos y pipeline de render ejecutado en segundo plano."""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .deadline import Deadline
from .exceptions import GPXRenderError
from .gpx_processor import ProcessedTrack
from .moments import MomentRenderer, PreparedMoment
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
    moments: list[PreparedMoment] = field(default_factory=list)
    #: Directorio temporal con los medios de los momentos (lo borra el pipeline al terminar).
    media_dir: Path | None = None

    def cinematic_duration_s(self, settings: Settings | None = None) -> float:
        """Segundos extra de intro+outro si la cinemática está activa."""
        s = settings or self.track.metrics  # type: ignore[arg-type]
        # se resuelve via settings global si no se pasa explícito
        from .config import get_settings as _gs

        cfg = settings if isinstance(settings, Settings) else _gs()
        if not cfg.cinematic_enabled:
            return 0.0
        return float(cfg.cinematic_intro_s + cfg.cinematic_outro_s)

    @property
    def total_duration_s(self) -> float:
        """Duración prevista del vídeo: ruta + momentos + intro/outro cinemáticos."""
        from .config import get_settings as _gs

        cfg = _gs()
        cinematic = 0.0
        if cfg.cinematic_enabled:
            cinematic = float(cfg.cinematic_intro_s + cfg.cinematic_outro_s)
        return self.track.video_duration_s + sum(m.duration_s for m in self.moments) + cinematic
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

    def list(self) -> list[Job]:
        """Todos los trabajos en memoria, del más reciente al más antiguo."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def purge_old(self, max_age_s: int = 86400, max_keep: int = 500) -> int:
        """A-5: purga trabajos terminados/fallidos antiguos para evitar fuga de memoria."""
        now = _now()
        with self._lock:
            # ordenar por updated_at descendente
            sorted_jobs = sorted(self._jobs.values(), key=lambda j: j.updated_at, reverse=True)
            to_keep = set()
            # conservar siempre los en curso y los más recientes
            for j in sorted_jobs:
                if j.status in (JobStatus.QUEUED, JobStatus.RENDERING, JobStatus.COMPILING):
                    to_keep.add(j.job_id)
            # conservar hasta max_keep recientes entre los terminados
            kept = 0
            for j in sorted_jobs:
                if j.job_id in to_keep:
                    continue
                age = (now - j.updated_at).total_seconds()
                if age > max_age_s and kept >= 0:
                    # más viejo que el límite → purgar si ya tenemos suficientes
                    continue
                if kept < max_keep:
                    to_keep.add(j.job_id)
                    kept += 1
            removed = 0
            for jid in list(self._jobs.keys()):
                if jid not in to_keep:
                    self._jobs.pop(jid, None)
                    removed += 1
            return removed

    def update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = _now()


class RenderPipeline:
    """Ejecuta render + compilación de un trabajo dentro de un directorio temporal propio.

    Los trabajos se despachan en un **executor propio y acotado**, no en el
    threadpool que Starlette comparte con todas las dependencias síncronas
    (40 hilos). Un render puede durar minutos: si ocupara ese pool, unas pocas
    decenas de subidas bloquearían también la autenticación y la API entera
    dejaría de responder. Aquí los trabajos esperan en una cola **sin consumir
    hilo**, y solo los ``max_concurrent_renders`` primeros se ejecutan a la vez.
    """

    def __init__(self, store: JobStore, settings: Settings, library: VideoLibrary) -> None:
        self._store = store
        self._library = library
        self._settings = settings
        self._workers = max(1, settings.max_concurrent_renders)
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future[None]] = set()
        self._futures_lock = threading.Lock()
        self._ensure_executor()

    # ------------------------------------------------------------------ #
    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Devuelve el executor de workers, creándolo si aún no existe o se cerró.

        Tras :meth:`shutdown` el pipeline debe poder volver a usarse: el servicio
        se reanuda en el mismo proceso al reiniciar la aplicación (y las pruebas
        abren y cierran el ``lifespan`` varias veces).
        """
        with self._futures_lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="gpxrender")
            return self._executor

    def submit(self, job_id: str) -> Future[None]:
        """Encola el trabajo y devuelve al instante (no bloquea el event loop).

        El futuro se registra en ``_futures`` para que :meth:`join` y
        :meth:`shutdown` sepan qué queda pendiente, y se retira al terminar.
        """
        future = self._ensure_executor().submit(self.run, job_id)
        with self._futures_lock:
            self._futures.add(future)
        future.add_done_callback(self._discard)
        return future

    def _discard(self, future: Future[None]) -> None:
        with self._futures_lock:
            self._futures.discard(future)

    def pending(self) -> list[Future[None]]:
        """Futuros aún no terminados (en cola o en ejecución)."""
        with self._futures_lock:
            return [f for f in self._futures if not f.done()]

    def join(self, timeout: float | None = None) -> bool:
        """Espera a que terminen todos los trabajos enviados.

        Returns:
            ``True`` si la cola quedó vacía; ``False`` si venció ``timeout``.
        """
        return not wait(self.pending(), timeout=timeout)[1]

    def shutdown(self, timeout_s: float = 5.0) -> None:
        """Detiene el pipeline al apagar el servicio.

        Los trabajos **en cola** se cancelan y conservan su estado ``queued`` (el
        cliente los sigue viendo en ``/api/v1/jobs`` y sabe que no llegaron a
        ejecutarse). Los **en curso** no se pueden interrumpir —Python no permite
        matar un hilo—, así que se les da ``timeout_s`` de gracia y después se
        marcan como fallidos y se retira su MP4 a medio escribir.
        """
        running = [f for f in self.pending() if f.running()]
        with self._futures_lock:
            executor, self._executor = self._executor, None  # el próximo submit crea uno nuevo
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if not wait(running, timeout=timeout_s)[1]:
            return  # terminaron todos a tiempo
        for job in self._store.list():
            if job.status in (JobStatus.RENDERING, JobStatus.COMPILING):
                logger.warning("Job %s interrumpido por el apagado del servidor", job.job_id)
                self._abandon(job)

    def _abandon(self, job: Job) -> None:
        """Marca un trabajo como fallido y retira su vídeo parcial."""
        try:
            self._store.update(job.job_id, status=JobStatus.FAILED, error="El servidor se detuvo durante el render.")
        except KeyError:  # el trabajo ya se había retirado del almacén
            return
        if job.video_path is not None:
            job.video_path.with_name(job.video_path.name + ".part").unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    def run(self, job_id: str) -> None:
        """Punto de entrada síncrono (lo invoca un worker del executor propio).

        Nunca propaga excepciones: cualquier fallo queda registrado en el trabajo.
        """
        job = self._store.get(job_id)
        if job is None:
            return
        settings = self._settings
        workdir = Path(tempfile.mkdtemp(prefix=f"gpxrender_{job_id}_"))
        try:
            # Plazo cooperativo: los motores lo consultan fotograma a fotograma (A-3).
            with Deadline(settings.render_timeout_s):
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
                    stall_timeout_s=settings.ffmpeg_stall_timeout_s,
                )
                frames_dir = workdir / "frames"
                moments_at: dict[int, list[PreparedMoment]] = {}
                for moment in job.moments:
                    moments_at.setdefault(min(moment.frame, payload.frame_count - 1), []).append(moment)
                moment_renderer = MomentRenderer(payload.width, payload.height, payload.fps, settings)

                def extra_frames(index: int, png: bytes) -> Iterator[bytes]:
                    """Tras el fotograma en el que cae un momento, intercala su foto o su clip."""
                    for moment in moments_at.get(index, ()):

                        yield from moment_renderer.frames(moment, png)

                # Configuración cinemática (intro/outro)
                use_cinematic = bool(
                    settings.cinematic_enabled and hasattr(orchestrator.engine, "render_intro")
                )
                intro_n = int(round(settings.cinematic_intro_s * payload.fps)) if use_cinematic else 0
                outro_n = int(round(settings.cinematic_outro_s * payload.fps)) if use_cinematic else 0
                cinematic_total = intro_n + payload.frame_count + outro_n

                with compiler:

                    def on_progress(done: int, total: int) -> None:
                        # Cada lote terminado se codifica y se borra: el disco temporal no crece.
                        compiler.feed(frames_dir, done, extra_frames)
                        if cinematic_total:
                            g_done = intro_n + done
                            self._store.update(job_id, progress=round(0.97 * g_done / cinematic_total, 3))
                        else:
                            self._store.update(job_id, progress=round(0.97 * done / total, 3))

                    if intro_n:
                        self._store.update(job_id, progress=0.02)
                        intro_dir = workdir / "intro_frames"
                        orchestrator.render_intro(payload, intro_dir, None)
                        for idx in range(intro_n):
                            p = intro_dir / f"frame_{idx:04d}.png"
                            compiler.write_png(p.read_bytes())
                            p.unlink(missing_ok=True)
                        try:
                            intro_dir.rmdir()
                        except OSError:
                            pass

                    orchestrator.render_frames(payload, workdir, on_progress, verify=False)
                    compiler.feed(frames_dir, payload.frame_count, extra_frames)  # motores que no informan del progreso
                    if outro_n:
                        outro_dir = workdir / "outro_frames"
                        orchestrator.render_outro(payload, outro_dir, None)
                        for idx in range(outro_n):
                            p = outro_dir / f"frame_{idx:04d}.png"
                            compiler.write_png(p.read_bytes())
                            p.unlink(missing_ok=True)
                        try:
                            outro_dir.rmdir()
                        except OSError:
                            pass
                    self._store.update(job_id, status=JobStatus.COMPILING, progress=0.97)
                    video = compiler.finish(payload.frame_count)
                    total_frames = compiler.frames_written
                self._library.register(
                    job_id,
                    video,
                    owner=job.owner,
                    title=job.title,
                    options=job.options,
                    metrics=job.track.metrics,
                    duration_s=total_frames / payload.fps,
                    width=payload.width,
                    height=payload.height,
                    moment_count=len(job.moments),
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
            if job.media_dir is not None:
                shutil.rmtree(job.media_dir, ignore_errors=True)

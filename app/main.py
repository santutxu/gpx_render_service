"""Aplicación FastAPI: ingesta de GPX y consulta de trabajos de render.

Arranque::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import PurePath

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import get_settings
from .exceptions import FileTooLargeError, GPXRenderError, UnsupportedFileError
from .gpx_processor import process_gpx
from .jobs import Job, JobStore, RenderPipeline
from .schemas import CameraStyle, ErrorResponse, JobInfo, JobStatus, RenderOptions, Resolution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
store = JobStore()
pipeline = RenderPipeline(store, settings)

_CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepara el directorio de salida y avisa pronto si falta FFmpeg."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which(settings.ffmpeg_binary) is None:
        logger.warning("FFmpeg ('%s') no está en el PATH: los renders fallarán.", settings.ffmpeg_binary)
    yield


app = FastAPI(
    title="GPX 3D Render Service",
    version=__version__,
    description="Convierte tracks GPX en vídeos animados (estilo Relive).",
    lifespan=lifespan,
)


@app.exception_handler(GPXRenderError)
async def domain_error_handler(_: Request, exc: GPXRenderError) -> JSONResponse:
    """Traduce las excepciones de dominio a respuestas HTTP homogéneas."""
    if exc.status_code >= 500:
        logger.error("%s: %s | %s", exc.error_code, exc.message, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.error_code, message=exc.message).model_dump(),
    )


async def _read_upload(file: UploadFile) -> bytes:
    """Lee la subida por bloques, abortando en cuanto supera el límite de tamaño."""
    name = PurePath(file.filename or "").name
    if not name.lower().endswith(".gpx"):
        raise UnsupportedFileError("Solo se admiten archivos con extensión .gpx.")
    chunks: list[bytes] = []
    total = 0
    try:
        while chunk := await file.read(_CHUNK):
            total += len(chunk)
            if total > settings.max_upload_bytes:
                raise FileTooLargeError(
                    f"El archivo supera el máximo de {settings.max_upload_bytes // (1024 * 1024)} MB."
                )
            chunks.append(chunk)
    finally:
        await file.close()  # libera el temporal en disco que Starlette crea para subidas grandes
    return b"".join(chunks)


def _job_info(job: Job, request: Request) -> JobInfo:
    done = job.status is JobStatus.COMPLETED
    return JobInfo(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        options=job.options,
        metrics=job.track.metrics,
        frame_count=job.track.frame_count,
        video_duration_s=job.track.video_duration_s,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        status_url=str(request.url_for("get_job", job_id=job.job_id)),
        video_url=str(request.url_for("download_video", job_id=job.job_id)) if done else None,
    )


def _get_job_or_404(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    return job


_ERRORS: dict[int | str, dict[str, object]] = {
    413: {"model": ErrorResponse, "description": "Archivo demasiado grande"},
    415: {"model": ErrorResponse, "description": "Extensión no admitida"},
    422: {"model": ErrorResponse, "description": "GPX inválido o parámetros incorrectos"},
}


@app.post("/api/v1/render-gpx", response_model=JobInfo, status_code=202, responses=_ERRORS, tags=["render"])
async def render_gpx(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Archivo .gpx con el track"),
    animation_speed: float = Form(1.0, ge=0.5, le=4.0, description="Multiplicador de velocidad de la animación"),
    resolution: Resolution = Form(Resolution.P1080, description="Resolución del vídeo"),
    camera_style: CameraStyle = Form(CameraStyle.FOLLOW, description="Estilo de cámara"),
) -> JobInfo:
    """Valida el GPX, calcula sus métricas y encola el render del vídeo.

    El GPX se valida y procesa **antes** de responder (un archivo corrupto
    devuelve 4xx de inmediato); el render y la compilación, que son costosos,
    se ejecutan en segundo plano. Consultar el progreso en ``status_url``.
    """
    options = RenderOptions(animation_speed=animation_speed, resolution=resolution, camera_style=camera_style)
    data = await _read_upload(file)
    # Parsing y filtrado son CPU-bound: fuera del event loop.
    track = await run_in_threadpool(process_gpx, data, options, settings)

    job = Job(job_id=uuid.uuid4().hex, options=options, track=track)
    store.add(job)
    background_tasks.add_task(pipeline.run, job.job_id)
    return _job_info(job, request)


@app.get("/api/v1/jobs/{job_id}", response_model=JobInfo, tags=["render"])
async def get_job(job_id: str, request: Request) -> JobInfo:
    """Estado, progreso y métricas de un trabajo."""
    return _job_info(_get_job_or_404(job_id), request)


@app.get("/api/v1/jobs/{job_id}/track", tags=["render"])
async def get_track(job_id: str) -> JSONResponse:
    """Trayectoria suavizada e interpolada (un punto por fotograma) en JSON."""
    job = _get_job_or_404(job_id)
    return JSONResponse({"fps": job.track.fps, "points": job.track.to_points()})


@app.get("/api/v1/jobs/{job_id}/video", response_class=FileResponse, tags=["render"])
async def download_video(job_id: str) -> FileResponse:
    """Descarga el MP4 resultante (409 si el trabajo aún no ha terminado)."""
    job = _get_job_or_404(job_id)
    if job.status is not JobStatus.COMPLETED or job.video_path is None or not job.video_path.is_file():
        raise HTTPException(status_code=409, detail=f"El vídeo no está disponible (estado: {job.status.value}).")
    return FileResponse(job.video_path, media_type="video/mp4", filename=f"{job_id}.mp4")


@app.get("/health", tags=["sistema"])
async def health() -> dict[str, object]:
    """Comprobación de vida y de dependencias externas."""
    return {
        "status": "ok",
        "version": __version__,
        "ffmpeg_available": shutil.which(settings.ffmpeg_binary) is not None,
        "render_engine": settings.render_engine,
    }

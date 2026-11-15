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

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import __version__
from .auth import SESSION_COOKIE, AuthenticationError, Principal, get_auth_manager, require_user
from .config import get_settings
from .exceptions import FileTooLargeError, GPXRenderError, UnsupportedFileError
from .gallery import GALLERY_HTML
from .gpx_processor import process_gpx
from .jobs import Job, JobStore, RenderPipeline
from .schemas import (
    CameraStyle,
    ErrorResponse,
    JobInfo,
    JobStatus,
    LoginRequest,
    RenderOptions,
    Resolution,
    StoredVideo,
    UserInfo,
    VideoInfo,
    VideoList,
)
from .video_library import VideoLibrary, VideoNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
store = JobStore()
auth = get_auth_manager()  # valida la configuración de claves al arrancar
library = VideoLibrary(settings.output_dir)
pipeline = RenderPipeline(store, settings, library)

_CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepara el directorio de salida y avisa pronto si falta FFmpeg."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if auth.generated_key_path:
        logger.warning("Acceso protegido con la clave de administrador guardada en %s", auth.generated_key_path)
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
        headers={"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None,
    )


def _title_from_filename(filename: str | None) -> str:
    """Título legible a partir del nombre del archivo subido (sin ruta, extensión ni caracteres de control)."""
    stem = PurePath((filename or "").replace("\\", "/")).stem
    clean = "".join(ch for ch in stem if ch.isprintable()).strip()
    return clean[:120] or "ruta"


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


def _get_job_or_404(job_id: str, user: Principal) -> Job:
    """Devuelve el trabajo si existe y pertenece al usuario; si no, 404 (no se revela que existe)."""
    job = store.get(job_id)
    if job is None or not user.can_access(job.owner):
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
    user: Principal = Depends(require_user),
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

    job = Job(job_id=uuid.uuid4().hex, options=options, track=track, title=_title_from_filename(file.filename), owner=user.name)
    store.add(job)
    background_tasks.add_task(pipeline.run, job.job_id)
    return _job_info(job, request)


@app.get("/api/v1/jobs/{job_id}", response_model=JobInfo, tags=["render"])
async def get_job(job_id: str, request: Request, user: Principal = Depends(require_user)) -> JobInfo:
    """Estado, progreso y métricas de un trabajo."""
    return _job_info(_get_job_or_404(job_id, user), request)


@app.get("/api/v1/jobs/{job_id}/track", tags=["render"])
async def get_track(job_id: str, user: Principal = Depends(require_user)) -> JSONResponse:
    """Trayectoria suavizada e interpolada (un punto por fotograma) en JSON."""
    job = _get_job_or_404(job_id, user)
    return JSONResponse({"fps": job.track.fps, "points": job.track.to_points()})


@app.get("/api/v1/jobs/{job_id}/video", response_class=FileResponse, tags=["render"])
async def download_video(job_id: str, user: Principal = Depends(require_user)) -> FileResponse:
    """Descarga el MP4 resultante (409 si el trabajo aún no ha terminado)."""
    job = _get_job_or_404(job_id, user)
    if job.status is not JobStatus.COMPLETED or job.video_path is None or not job.video_path.is_file():
        raise HTTPException(status_code=409, detail=f"El vídeo no está disponible (estado: {job.status.value}).")
    return FileResponse(job.video_path, media_type="video/mp4", filename=f"{job_id}.mp4")


# --------------------------------------------------------------------------- #
# Autenticación
# --------------------------------------------------------------------------- #
@app.post("/api/v1/auth/login", response_model=UserInfo, tags=["autenticación"])
async def login(credentials: LoginRequest, request: Request, response: Response) -> UserInfo:
    """Canjea una clave de API por una cookie de sesión (para la galería web y la etiqueta ``<video>``)."""
    if not auth.enabled:
        return UserInfo(name="anonymous", is_admin=True)
    principal = auth.authenticate_key(credentials.api_key)
    if principal is None:
        logger.warning("Inicio de sesión fallido desde %s", request.client.host if request.client else "?")
        raise AuthenticationError("Clave de API no válida.")
    response.set_cookie(
        SESSION_COOKIE,
        auth.create_session(principal),
        max_age=settings.session_ttl_s,
        httponly=True,  # inaccesible desde JavaScript
        samesite="strict",  # no se envía en peticiones originadas en otros sitios (CSRF)
        secure=request.url.scheme == "https",
        path="/",
    )
    return UserInfo(name=principal.name, is_admin=principal.is_admin)


@app.post("/api/v1/auth/logout", status_code=204, tags=["autenticación"])
async def logout() -> Response:
    """Cierra la sesión del navegador."""
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/api/v1/auth/me", response_model=UserInfo, tags=["autenticación"])
async def whoami(user: Principal = Depends(require_user)) -> UserInfo:
    """Usuario al que corresponden las credenciales enviadas."""
    return UserInfo(name=user.name, is_admin=user.is_admin)


# --------------------------------------------------------------------------- #
# Biblioteca de vídeos
# --------------------------------------------------------------------------- #
_NOT_FOUND: dict[int | str, dict[str, object]] = {404: {"model": ErrorResponse, "description": "Vídeo no encontrado"}}


def _video_info(video: StoredVideo, request: Request) -> VideoInfo:
    return VideoInfo(
        **video.model_dump(),
        stream_url=str(request.url_for("stream_video", video_id=video.video_id)),
        download_url=str(request.url_for("download_library_video", video_id=video.video_id)),
    )


def _owned_video(video_id: str, user: Principal) -> StoredVideo:
    """Metadatos del vídeo si el usuario tiene acceso; 404 en caso contrario."""
    video = library.get(video_id)
    if not user.can_access(video.owner):
        raise VideoNotFoundError("Vídeo no encontrado.")
    return video


def _download_name(video: StoredVideo) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_. " else "_" for ch in video.title).strip() or video.video_id
    return f"{safe}.mp4"


@app.get("/api/v1/videos", response_model=VideoList, tags=["vídeos"])
async def list_videos(
    request: Request,
    limit: int = Query(50, ge=1, le=200, description="Tamaño de página"),
    offset: int = Query(0, ge=0, description="Vídeos a saltar"),
    user: Principal = Depends(require_user),
) -> VideoList:
    """Lista los vídeos del usuario (todos, si es administrador), del más reciente al más antiguo."""
    videos = [v for v in await run_in_threadpool(library.list) if user.can_access(v.owner)]
    page = videos[offset : offset + limit]
    return VideoList(total=len(videos), limit=limit, offset=offset, items=[_video_info(v, request) for v in page])


@app.get("/api/v1/videos/{video_id}", response_model=VideoInfo, responses=_NOT_FOUND, tags=["vídeos"])
async def get_video(video_id: str, request: Request, user: Principal = Depends(require_user)) -> VideoInfo:
    """Metadatos de un vídeo: métricas de la ruta, tamaño, duración y enlaces."""
    return _video_info(_owned_video(video_id, user), request)


@app.get("/api/v1/videos/{video_id}/stream", response_class=FileResponse, responses=_NOT_FOUND, tags=["vídeos"])
async def stream_video(video_id: str, user: Principal = Depends(require_user)) -> FileResponse:
    """Reproduce el vídeo en el navegador (``inline``; admite ``Range`` para saltar por la línea de tiempo)."""
    video = _owned_video(video_id, user)
    return FileResponse(
        library.path(video_id), media_type="video/mp4", filename=_download_name(video), content_disposition_type="inline"
    )


@app.get("/api/v1/videos/{video_id}/download", response_class=FileResponse, responses=_NOT_FOUND, tags=["vídeos"])
async def download_library_video(video_id: str, user: Principal = Depends(require_user)) -> FileResponse:
    """Descarga el vídeo como archivo adjunto, con el nombre del GPX original."""
    video = _owned_video(video_id, user)
    return FileResponse(library.path(video_id), media_type="video/mp4", filename=_download_name(video))


@app.delete("/api/v1/videos/{video_id}", status_code=204, responses=_NOT_FOUND, tags=["vídeos"])
async def delete_video(video_id: str, user: Principal = Depends(require_user)) -> Response:
    """Borra definitivamente el vídeo y sus metadatos (solo su propietario o un administrador)."""
    _owned_video(video_id, user)
    await run_in_threadpool(library.delete, video_id)
    store.remove(video_id)  # el trabajo asociado deja de existir
    return Response(status_code=204)


@app.get("/videos", response_class=HTMLResponse, include_in_schema=False)
async def gallery() -> HTMLResponse:
    """Galería web para ver, descargar y borrar los vídeos."""
    return HTMLResponse(GALLERY_HTML)


@app.get("/health", tags=["sistema"])
async def health() -> dict[str, object]:
    """Comprobación de vida y de dependencias externas."""
    return {
        "status": "ok",
        "version": __version__,
        "ffmpeg_available": shutil.which(settings.ffmpeg_binary) is not None,
        "render_engine": settings.render_engine,
    }

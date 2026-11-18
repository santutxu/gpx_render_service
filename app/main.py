"""Aplicación FastAPI: ingesta de GPX y consulta de trabajos de render.

Arranque::

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePath

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .auth import (
    SESSION_COOKIE,
    AuthenticationError,
    ForbiddenError,
    Principal,
    get_auth_manager,
    require_admin,
    require_user,
)
from .config import get_settings
from .exceptions import FileTooLargeError, GPXRenderError, QuotaExceededError, ServerBusyError, UnsupportedFileError
from .gpx_processor import process_gpx
from .jobs import Job, JobStore, RenderPipeline
from .moments import InvalidMomentError, PreparedMoment, parse_moment_specs, prepare_moments
from .schemas import (
    AdminUserCreate,
    AdminUserInfo,
    AdminUserUpdate,
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyInfo,
    AuthConfig,
    PasswordChange,
    RegisterRequest,
    CameraStyle,
    ErrorResponse,
    JobInfo,
    JobStatus,
    LoginRequest,
    Orientation,
    RenderOptions,
    Resolution,
    StoredVideo,
    UserInfo,
    VideoInfo,
    VideoList,
)
from .terrain import fetch_tile_bytes
from .users import TooManyKeysError, UsernameTakenError, hash_password, verify_password
from .video_library import VideoLibrary, VideoNotFoundError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()
store = JobStore()
auth = get_auth_manager()  # valida la configuración de claves al arrancar
library = VideoLibrary(settings.output_dir)
pipeline = RenderPipeline(store, settings, library)

_CHUNK = 1024 * 1024
#: Antigüedad a partir de la cual un ``.mp4.part`` se considera huérfano de un arranque anterior.
#: Se toleran los recientes para no borrar el vídeo que esté escribiendo otra réplica del servicio.
_ORPHAN_AFTER_S = 24 * 3600


def _sweep_orphan_partials(directory: Path) -> int:
    """Borra los MP4 a medio escribir que dejó un apagado brusco.

    El compilador escribe en ``<id>.mp4.part`` y lo renombra al terminar, así que
    un ``.part`` viejo es siempre basura: nunca se sirve ni se lista.
    """
    if not directory.is_dir():
        return 0
    cutoff = time.time() - _ORPHAN_AFTER_S
    removed = 0
    for partial in directory.glob("*.mp4.part"):
        try:
            if partial.stat().st_mtime < cutoff:
                partial.unlink()  # Path.unlink() no devuelve nada: no sirve para contar
                removed += 1
        except OSError as exc:  # archivo en uso o sin permisos: se reintentará en el próximo arranque
            logger.warning("No se pudo borrar el vídeo parcial %s: %s", partial.name, exc)
    return removed


_PURGE_INTERVAL_S = 3600  # A-5: purga periódica de trabajos antiguos


def _purge_loop(stop: threading.Event) -> None:
    """Hilo en segundo plano: cada _PURGE_INTERVAL_S purga trabajos terminados antiguos."""
    while not stop.wait(_PURGE_INTERVAL_S):
        try:
            removed = store.purge_old()
            if removed:
                logger.info("Purgados %d trabajo(s) antiguos de memoria.", removed)
        except Exception:  # pragma: no cover
            logger.exception("Error en purga periódica de trabajos")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepara el directorio de salida, avisa si falta FFmpeg y detiene el pipeline al cerrar."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which(settings.ffmpeg_binary) is None:
        # intentar fallbacks persistentes antes de avisar
        found = None
        for cand in (Path.home() / "bin" / "ffmpeg", Path("/home/user/bin/ffmpeg")):
            if cand.is_file():
                found = cand
                break
        if found is not None:
            logger.info("Usando FFmpeg en %s", found)
        else:
            try:
                import imageio_ffmpeg  # type: ignore[import-not-found]

                if Path(imageio_ffmpeg.get_ffmpeg_exe()).is_file():
                    logger.info("Usando FFmpeg de imageio-ffmpeg: %s", imageio_ffmpeg.get_ffmpeg_exe())
                else:
                    raise FileNotFoundError
            except Exception:
                logger.warning("FFmpeg ('%s') no está en el PATH: los renders fallarán.", settings.ffmpeg_binary)
    if orphans := await run_in_threadpool(_sweep_orphan_partials, settings.output_dir):
        logger.info("Borrados %d vídeo(s) parciales de arranques anteriores.", orphans)
    stop = threading.Event()
    t = threading.Thread(target=_purge_loop, args=(stop,), name="gpxrender-purge", daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        # Los renders ya no viven en el threadpool de Starlette: hay que detenerlos aquí.
        await run_in_threadpool(pipeline.shutdown, settings.shutdown_grace_s)


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


@app.exception_handler(HTTPException)
async def http_error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """B-8: normaliza HTTPException a {error, message} manteniendo compatibilidad con {detail}."""
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    code_map = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "conflict", 413: "payload_too_large", 415: "unsupported_media", 422: "validation_error", 429: "too_many_requests", 502: "bad_gateway", 503: "service_unavailable"}
    # Devolver ambos formatos para no romper tests/frontend existentes
    payload = ErrorResponse(error=code_map.get(exc.status_code, "http_error"), message=detail).model_dump()
    payload["detail"] = detail
    return JSONResponse(status_code=exc.status_code, content=payload, headers=exc.headers)


def _sanitize_title(raw: str | None, fallback_filename: str | None = None) -> str:
    """Normaliza un título aportado por el usuario o derivado del nombre del fichero."""
    if raw is not None:
        candidate = "".join(ch for ch in raw if ch.isprintable()).strip()
        candidate = " ".join(candidate.split())
        if candidate:
            return candidate[:120]
    stem = PurePath((fallback_filename or "").replace("\\", "/")).stem
    clean = "".join(ch for ch in stem if ch.isprintable()).strip()
    clean = " ".join(clean.split())
    return clean[:120] or "ruta"


def _title_from_filename(filename: str | None) -> str:
    """Compat: título derivado solo del nombre de fichero."""
    return _sanitize_title(None, filename)


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


async def _save_moment_file(file: UploadFile, dest: Path) -> Path:
    """Guarda en disco el adjunto de un momento, por bloques y con límite de tamaño."""
    total = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(_CHUNK):
                total += len(chunk)
                if total > settings.max_moment_file_bytes:
                    raise FileTooLargeError(
                        f"Cada foto o clip puede ocupar como máximo {settings.max_moment_file_bytes // (1024 * 1024)} MB."
                    )
                out.write(chunk)
    finally:
        await file.close()
    if total == 0:
        raise InvalidMomentError("Uno de los archivos adjuntos está vacío.")
    return dest


def _job_info(job: Job, request: Request) -> JobInfo:
    done = job.status is JobStatus.COMPLETED
    return JobInfo(
        job_id=job.job_id,
        title=job.title,
        status=job.status,
        progress=job.progress,
        options=job.options,
        metrics=job.track.metrics,
        frame_count=job.track.frame_count,
        video_duration_s=round(job.total_duration_s, 2),
        moment_count=len(job.moments),
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


_IN_PROGRESS = (JobStatus.QUEUED, JobStatus.RENDERING, JobStatus.COMPILING)


def _check_quotas(user: Principal) -> None:
    """Rechaza el trabajo antes de procesar nada si el usuario o la cola han llegado a su límite.

    Sin estas cuotas, una sola cuenta puede acaparar los workers de render y
    llenar ``GPX_OUTPUT_DIR`` indefinidamente. Se comprueban **antes** de leer la
    subida: rechazar es barato y no consume ancho de banda ni CPU.

    Raises:
        ServerBusyError: 503, la cola global está llena.
        QuotaExceededError: 429, el usuario ha agotado sus trabajos o su espacio.
    """
    jobs = store.list()
    if settings.max_queued_renders > 0:
        in_flight = sum(1 for job in jobs if job.status in _IN_PROGRESS)
        if in_flight >= settings.max_queued_renders:
            raise ServerBusyError(
                f"El servicio tiene {in_flight} vídeos en proceso. Vuelve a intentarlo en unos minutos."
            )
    if settings.max_jobs_per_user > 0:
        mine = sum(
            1
            for job in jobs
            if job.status in _IN_PROGRESS and job.owner and job.owner.lower() == user.name.lower()
        )
        if mine >= settings.max_jobs_per_user:
            raise QuotaExceededError(
                f"Ya tienes {mine} vídeos en proceso; el máximo son {settings.max_jobs_per_user}. "
                "Espera a que terminen o bórralos."
            )
    if settings.max_storage_bytes_per_user > 0 and user.user_id is not None:
        used = library.storage_bytes(user.name)
        if used >= settings.max_storage_bytes_per_user:
            limit = settings.max_storage_bytes_per_user / 1e9
            raise QuotaExceededError(
                f"Has alcanzado el límite de espacio ({used / 1e9:.2f} GB de {limit:.2f} GB). Borra algún vídeo."
            )


_ERRORS: dict[int | str, dict[str, object]] = {
    413: {"model": ErrorResponse, "description": "Archivo demasiado grande"},
    415: {"model": ErrorResponse, "description": "Extensión no admitida"},
    422: {"model": ErrorResponse, "description": "GPX inválido o parámetros incorrectos"},
    429: {"model": ErrorResponse, "description": "Cuota de trabajos o de espacio agotada"},
    503: {"model": ErrorResponse, "description": "Cola de render llena"},
}


@app.post("/api/v1/render-gpx", response_model=JobInfo, status_code=202, responses=_ERRORS, tags=["render"])
async def render_gpx(
    request: Request,
    file: UploadFile = File(..., description="Archivo .gpx con el track"),
    title: str | None = Form(None, max_length=120, description="Nombre del vídeo (opcional, máx. 120 caracteres)"),
    animation_speed: float = Form(1.0, ge=0.5, le=4.0, description="Multiplicador de velocidad de la animación"),
    resolution: Resolution = Form(Resolution.P1080, description="Resolución del vídeo"),
    orientation: Orientation = Form(Orientation.HORIZONTAL, description="Orientación del lienzo: horizontal (16:9) o vertical (9:16) para redes sociales"),
    camera_style: CameraStyle = Form(CameraStyle.FOLLOW, description="Estilo de cámara"),
    moments: str | None = Form(
        None,
        description='Momentos en JSON: [{"lat": 43.2, "lon": -2.9, "caption": "Cima", "file_index": 0}]. '
        "'file_index' es la posición del adjunto en 'moment_files'; 'position' (0-1) es opcional.",
    ),
    moment_files: list[UploadFile] = File(default=[], description="Fotos (JPG, PNG, WebP) o clips (MP4, MOV, WebM)"),
    user: Principal = Depends(require_user),
) -> JobInfo:
    """Valida el GPX, calcula sus métricas y encola el render del vídeo.

    Opcionalmente se pueden adjuntar **momentos**: fotos o clips anclados a un punto de la
    ruta, que se intercalan en el vídeo cuando el recorrido llega a ese punto.

    El GPX se valida y procesa **antes** de responder (un archivo corrupto
    devuelve 4xx de inmediato); el render y la compilación, que son costosos,
    se ejecutan en segundo plano. Consultar el progreso en ``status_url``.

    Raises:
        QuotaExceededError: 429 si el usuario tiene demasiados trabajos en curso
            o ha agotado su espacio (``GPX_MAX_JOBS_PER_USER``, ``GPX_MAX_STORAGE_BYTES_PER_USER``).
        ServerBusyError: 503 si la cola global está llena (``GPX_MAX_QUEUED_RENDERS``).
    """
    _check_quotas(user)
    options = RenderOptions(animation_speed=animation_speed, resolution=resolution, camera_style=camera_style, orientation=orientation)
    data = await _read_upload(file)
    # Parsing y filtrado son CPU-bound: fuera del event loop.
    track = await run_in_threadpool(process_gpx, data, options, settings)

    specs = parse_moment_specs(moments, len(moment_files), settings)
    prepared: list[PreparedMoment] = []
    media_dir: Path | None = None
    if specs:
        media_dir = Path(tempfile.mkdtemp(prefix="gpxmoments_"))
        try:
            raw_files = {
                spec.file_index: await _save_moment_file(moment_files[spec.file_index], media_dir / f"raw_{spec.file_index}")
                for spec in specs
            }
            prepared = await run_in_threadpool(prepare_moments, specs, raw_files, media_dir, track, settings)
        except BaseException:
            shutil.rmtree(media_dir, ignore_errors=True)  # nada de adjuntos huérfanos si la validación falla
            raise

    job = Job(
        job_id=uuid.uuid4().hex,
        options=options,
        track=track,
        title=_sanitize_title(title, file.filename),
        owner=user.name,
        moments=prepared,
        media_dir=media_dir,
    )
    store.add(job)
    # Al executor propio del pipeline: encolar no consume hilo ni bloquea el event loop.
    pipeline.submit(job.job_id)
    return _job_info(job, request)


@app.get("/api/v1/jobs", response_model=list[JobInfo], tags=["render"])
def list_jobs(request: Request, user: Principal = Depends(require_user)) -> list[JobInfo]:
    """Trabajos del usuario desde el último arranque (los terminados están además en ``/api/v1/videos``)."""
    return [_job_info(job, request) for job in store.list() if user.can_access(job.owner)]


@app.delete("/api/v1/jobs/{job_id}", status_code=204, tags=["render"])
def dismiss_job(job_id: str, user: Principal = Depends(require_user)) -> Response:
    """Descarta del listado un trabajo fallido o terminado (no borra el vídeo). 409 si sigue en curso."""
    job = _get_job_or_404(job_id, user)
    if job.status not in (JobStatus.FAILED, JobStatus.COMPLETED):
        raise HTTPException(status_code=409, detail="El trabajo sigue en curso.")
    store.remove(job_id)
    return Response(status_code=204)


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
# Autenticación y cuentas
# --------------------------------------------------------------------------- #
def _user_info(user: Principal) -> UserInfo:
    return UserInfo(name=user.name, is_admin=user.is_admin, has_account=user.user_id is not None)


def _start_session(response: Response, request: Request, principal: Principal) -> None:
    """Añade a la respuesta la cookie de sesión firmada."""
    samesite = settings.cookie_samesite if settings.cookie_samesite in ("strict", "lax", "none") else "strict"
    response.set_cookie(
        SESSION_COOKIE,
        auth.create_session(principal),
        max_age=settings.session_ttl_s,
        httponly=True,  # inaccesible desde JavaScript
        samesite=samesite,  # strict: no se envía en peticiones originadas en otros sitios (CSRF)
        secure=settings.cookie_secure or samesite == "none" or request.url.scheme == "https",
        path="/",
    )


def _client_ip(request: Request) -> str:
    # B-3: detrás de nginx/Traefik la IP real viene en X-Forwarded-For
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # primera IP de la lista es la del cliente original
        return forwarded.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "?"


def _account_id(user: Principal) -> int:
    """Id de la cuenta registrada; 403 para claves estáticas de entorno o acceso anónimo."""
    if user.user_id is None:
        raise ForbiddenError("Esta operación requiere una cuenta de usuario registrada.")
    return user.user_id


@app.get("/api/v1/auth/config", response_model=AuthConfig, tags=["autenticación"])
async def auth_config() -> AuthConfig:
    """Indica a la web si hay que iniciar sesión y si se pueden crear cuentas."""
    return AuthConfig(
        auth_enabled=auth.enabled,
        registration_open=settings.allow_registration,
        map_attribution=settings.imagery_attribution,
        max_moments=settings.max_moments,
    )


@app.post("/api/v1/auth/register", response_model=UserInfo, status_code=201, tags=["autenticación"])
def register(data: RegisterRequest, request: Request, response: Response) -> UserInfo:
    """Crea una cuenta e inicia sesión. La primera cuenta del sistema es administradora."""
    # B-1: limitar intentos de registro por IP (mismo limitador que login)
    bucket = f"register|{_client_ip(request)}"
    auth.limiter.check(bucket)
    if not settings.allow_registration:
        raise ForbiddenError("El registro de nuevas cuentas está desactivado.")
    if auth.is_name_reserved(data.username):
        # cuenta como intento fallido para el limitador
        auth.limiter.record_failure(bucket)
        raise HTTPException(status_code=409, detail="Ese nombre de usuario no está disponible.")
    try:
        user = auth.users.create_user(data.username, hash_password(data.password))
    except UsernameTakenError:
        auth.limiter.record_failure(bucket)
        raise HTTPException(status_code=409, detail="Ese nombre de usuario no está disponible.") from None
    auth.limiter.reset(bucket)
    logger.info("Cuenta creada: %s%s", user.username, " (administrador)" if user.is_admin else "")
    principal = Principal(user.username, user.is_admin, user.id)
    _start_session(response, request, principal)
    return _user_info(principal)


@app.post("/api/v1/auth/login", response_model=UserInfo, tags=["autenticación"])
def login(credentials: LoginRequest, request: Request, response: Response) -> UserInfo:
    """Inicia sesión con usuario y contraseña (o con una clave de API) y devuelve una cookie de sesión.

    Tras 5 intentos fallidos en 5 minutos desde la misma IP para el mismo usuario, responde 429.
    """
    if not auth.enabled:
        return UserInfo(name="anonymous", is_admin=True)
    bucket = f"{_client_ip(request)}|{(credentials.username or 'api-key').lower()}"
    auth.limiter.check(bucket)
    if credentials.api_key:
        principal = auth.authenticate_key(credentials.api_key)
    elif credentials.username and credentials.password:
        principal = auth.authenticate_password(credentials.username, credentials.password)
    else:
        raise HTTPException(status_code=422, detail="Indica usuario y contraseña, o una clave de API.")
    if principal is None:
        auth.limiter.record_failure(bucket)
        logger.warning("Inicio de sesión fallido desde %s", _client_ip(request))
        raise AuthenticationError("Usuario o contraseña incorrectos.")
    auth.limiter.reset(bucket)
    _start_session(response, request, principal)
    return _user_info(principal)


@app.post("/api/v1/auth/logout", status_code=204, tags=["autenticación"])
async def logout() -> Response:
    """Cierra la sesión del navegador."""
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.get("/api/v1/auth/me", response_model=UserInfo, tags=["autenticación"])
def whoami(user: Principal = Depends(require_user)) -> UserInfo:
    """Usuario al que corresponden las credenciales enviadas."""
    return _user_info(user)


@app.put("/api/v1/auth/password", status_code=204, tags=["autenticación"])
def change_password(data: PasswordChange, request: Request, user: Principal = Depends(require_user)) -> Response:
    """Cambia la contraseña de la cuenta (requiere la actual). Cierra el resto de sesiones abiertas."""
    user_id = _account_id(user)
    account = auth.users.get_user(user.name)
    if account is None or not verify_password(data.current_password, account.password_hash):
        raise ForbiddenError("La contraseña actual no es correcta.")
    auth.users.set_password(user_id, hash_password(data.new_password))
    response = Response(status_code=204)
    _start_session(response, request, user)  # la sesión actual se renueva; las demás quedan invalidadas
    return response


# --------------------------------------------------------------------------- #
# Administración de usuarios
# --------------------------------------------------------------------------- #
def _admin_user_info(account_id: int) -> AdminUserInfo:
    # C-3: no escanear toda la biblioteca para un solo usuario
    account = auth.users.get_user_by_id(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    # buscar claves
    key_count = 0
    for acc, kc in auth.users.list_users():
        if acc.id == account_id:
            key_count = kc
            break
    # solo vídeos de este usuario
    own = [v for v in library.list() if v.owner and v.owner.lower() == account.username.lower()]
    return AdminUserInfo(
        id=account.id,
        username=account.username,
        is_admin=account.is_admin,
        created_at=account.created_at,
        video_count=len(own),
        api_key_count=key_count,
        storage_bytes=sum(v.size_bytes for v in own),
    )


def _all_users() -> list[AdminUserInfo]:
    # B-5: indexar vídeos por propietario para evitar O(usuarios × vídeos)
    videos = library.list()
    by_owner: dict[str, list] = {}
    for v in videos:
        if v.owner:
            by_owner.setdefault(v.owner.lower(), []).append(v)
    result = []
    for account, key_count in auth.users.list_users():
        own = by_owner.get(account.username.lower(), [])
        result.append(
            AdminUserInfo(
                id=account.id,
                username=account.username,
                is_admin=account.is_admin,
                created_at=account.created_at,
                video_count=len(own),
                api_key_count=key_count,
                storage_bytes=sum(v.size_bytes for v in own),
            )
        )
    return result


@app.get("/api/v1/admin/users", response_model=list[AdminUserInfo], tags=["administración"])
def admin_list_users(_: Principal = Depends(require_admin)) -> list[AdminUserInfo]:
    """Todas las cuentas, con su número de vídeos, claves y espacio ocupado."""
    return _all_users()


@app.post("/api/v1/admin/users", response_model=AdminUserInfo, status_code=201, tags=["administración"])
def admin_create_user(data: AdminUserCreate, admin: Principal = Depends(require_admin)) -> AdminUserInfo:
    """Crea una cuenta (funciona aunque el registro público esté cerrado)."""
    if auth.is_name_reserved(data.username):
        raise HTTPException(status_code=409, detail="Ese nombre de usuario no está disponible.")
    try:
        account = auth.users.create_user(data.username, hash_password(data.password), is_admin=data.is_admin)
    except UsernameTakenError:
        raise HTTPException(status_code=409, detail="Ese nombre de usuario no está disponible.") from None
    logger.info("Cuenta %s creada por el administrador %s", account.username, admin.name)
    return _admin_user_info(account.id)


@app.patch("/api/v1/admin/users/{user_id}", response_model=AdminUserInfo, tags=["administración"])
def admin_update_user(user_id: int, data: AdminUserUpdate, admin: Principal = Depends(require_admin)) -> AdminUserInfo:
    """Cambia el rol de una cuenta o restablece su contraseña (lo que cierra sus sesiones)."""
    account = auth.users.get_user_by_id(user_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    if data.is_admin is not None and data.is_admin != account.is_admin:
        if account.id == admin.user_id:
            raise HTTPException(status_code=409, detail="No puedes quitarte a ti mismo el rol de administrador.")
        auth.users.set_admin(user_id, data.is_admin)
        logger.info("%s cambia el rol de %s: admin=%s", admin.name, account.username, data.is_admin)
    if data.new_password is not None:
        auth.users.set_password(user_id, hash_password(data.new_password))
        logger.info("%s restablece la contraseña de %s", admin.name, account.username)
    return _admin_user_info(user_id)


@app.delete("/api/v1/admin/users/{user_id}", status_code=204, tags=["administración"])
def admin_delete_user(user_id: int, admin: Principal = Depends(require_admin)) -> Response:
    """Elimina una cuenta junto con sus claves de API y todos sus vídeos."""
    account = auth.users.get_user_by_id(user_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    if account.id == admin.user_id:
        raise HTTPException(status_code=409, detail="No puedes eliminar tu propia cuenta.")
    def owned(owner: str | None) -> bool:
        return owner is not None and owner.lower() == account.username.lower()

    jobs = [job for job in store.list() if owned(job.owner)]
    if any(job.status not in (JobStatus.COMPLETED, JobStatus.FAILED) for job in jobs):
        raise HTTPException(status_code=409, detail="El usuario tiene vídeos en proceso. Inténtalo cuando terminen.")
    for job in jobs:
        store.remove(job.job_id)
    for video in library.list():
        if owned(video.owner):
            try:
                library.delete(video.video_id)
            except VideoNotFoundError:
                pass
    auth.users.delete_user(user_id)
    logger.info("%s elimina la cuenta %s", admin.name, account.username)
    return Response(status_code=204)


# --------------------------------------------------------------------------- #
# Teselas del mapa 2D
# --------------------------------------------------------------------------- #
@app.get("/api/v1/tiles/{z}/{x}/{y}", tags=["mapa"], response_class=Response)
def map_tile(z: int, x: int, y: int, _: Principal = Depends(require_user)) -> Response:
    """Tesela de imagen satélite para el mapa de la web (servida desde la caché del servidor)."""
    if not (0 <= z <= 19 and 0 <= x < 2**z and 0 <= y < 2**z):
        raise HTTPException(status_code=404, detail="Tesela inexistente.")
    try:
        data = fetch_tile_bytes(settings.imagery_url.format(z=z, x=x, y=y), settings.tile_cache_dir, settings.tile_timeout_s, 2)
    except GPXRenderError:
        raise HTTPException(status_code=502, detail="El servidor de mapas no responde.") from None
    media_type = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
    return Response(data, media_type=media_type, headers={"Cache-Control": "private, max-age=604800"})


# --------------------------------------------------------------------------- #
# Claves de API
# --------------------------------------------------------------------------- #
@app.get("/api/v1/api-keys", response_model=list[ApiKeyInfo], tags=["claves de API"])
def list_api_keys(user: Principal = Depends(require_user)) -> list[ApiKeyInfo]:
    """Claves de API del usuario (sin su valor)."""
    return [ApiKeyInfo(**vars(k)) for k in auth.users.list_api_keys(_account_id(user))]


@app.post("/api/v1/api-keys", response_model=ApiKeyCreated, status_code=201, tags=["claves de API"])
def create_api_key(data: ApiKeyCreate, user: Principal = Depends(require_user)) -> ApiKeyCreated:
    """Crea una clave de API. Su valor completo solo se devuelve en esta respuesta."""
    try:
        info, key = auth.users.create_api_key(_account_id(user), data.name.strip() or "clave")
    except TooManyKeysError:
        raise HTTPException(status_code=409, detail="Has alcanzado el máximo de claves. Revoca alguna.") from None
    return ApiKeyCreated(**vars(info), key=key)


@app.delete("/api/v1/api-keys/{key_id}", status_code=204, tags=["claves de API"])
def revoke_api_key(key_id: int, user: Principal = Depends(require_user)) -> Response:
    """Revoca una clave: deja de funcionar inmediatamente."""
    if not auth.users.delete_api_key(_account_id(user), key_id):
        raise HTTPException(status_code=404, detail="Clave no encontrada.")
    return Response(status_code=204)


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


# --------------------------------------------------------------------------- #
# Seguridad: cabeceras CSP en todas las respuestas (C-9)
# --------------------------------------------------------------------------- #
_STATIC_DIR = Path(__file__).parent / "static"
# Sin scripts ni estilos en línea ni recursos externos: una inyección de HTML no puede ejecutar código.
# frame-ancestors evita que la app se incruste en iframes de otros orígenes.
_CSP = (
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'self'"
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    # CSP en todo (incluido /static/*) + anti-clickjacking
    response.headers.setdefault("Content-Security-Policy", _CSP)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    return response


@app.get("/", include_in_schema=False)
async def web_app() -> FileResponse:
    """Aplicación web: acceso, panel de rutas, subida de GPX y claves de API."""
    return FileResponse(
        _STATIC_DIR / "index.html",
        media_type="text/html",
        headers={"Content-Security-Policy": _CSP, "Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/videos", include_in_schema=False)
async def legacy_gallery() -> RedirectResponse:
    """Antigua galería: ahora forma parte de la aplicación web."""
    return RedirectResponse("/", status_code=307)


app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/health", tags=["sistema"])
async def health() -> dict[str, object]:
    """Comprobación de vida y de dependencias externas."""
    return {
        "status": "ok",
        "version": __version__,
        "ffmpeg_available": shutil.which(settings.ffmpeg_binary) is not None,
        "render_engine": settings.render_engine,
    }

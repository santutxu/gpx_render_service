"""Orquestación del motor gráfico headless.

El backend no dibuja en 3D: prepara un *payload* JSON (contrato
:class:`~app.schemas.RenderPayload`) y delega en un motor intercambiable que
debe dejar ``frame_0000.png … frame_NNNN.png`` en un directorio.

Motores incluidos:
  * :class:`~app.terrain_engine.Terrain3DEngine` – 3D real: relieve + imagen satélite (OpenGL headless).
  * :class:`NodeThreeJsEngine` – lanza un worker Node.js/Three.js por ``subprocess``.
  * :class:`SimulatedEngine` – genera fotogramas 2D con Pillow; simula el flujo
    completo sin GPU ni Node (desarrollo, CI y pruebas de integración).
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import Settings, get_settings
from .exceptions import GPXRenderError, RenderEngineError
from .gpx_processor import ProcessedTrack
from .schemas import CameraConfig, CameraStyle, FramePoint, RenderOptions, RenderPayload

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
FRAME_NAME = "frame_{:04d}.png"
FRAME_REGEX = re.compile(r"^frame_(\d{4,})\.png$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RenderEngine(Protocol):
    """Interfaz que debe cumplir cualquier motor gráfico."""

    name: str

    def render(
        self, payload: RenderPayload, payload_path: Path, frames_dir: Path, on_progress: ProgressCallback | None
    ) -> None:
        """Genera todos los fotogramas del payload en ``frames_dir``.

        Raises:
            RenderEngineError: si el motor falla.
        """
        ...


# --------------------------------------------------------------------------- #
# Motor real: worker Node.js / Three.js
# --------------------------------------------------------------------------- #
class NodeThreeJsEngine:
    """Invoca un worker headless de Node.js (Three.js + headless-gl o Puppeteer).

    Contrato de línea de comandos del worker::

        node worker.js --payload <payload.json> --out <frames_dir>

    El worker debe escribir ``frame_%04d.png`` (base 0) y terminar con código 0.
    """

    name = "node"

    def __init__(self, node_binary: str, worker_script: str, timeout_s: int) -> None:
        self._node = node_binary
        self._script = Path(worker_script)
        self._timeout = timeout_s

    def render(
        self, payload: RenderPayload, payload_path: Path, frames_dir: Path, on_progress: ProgressCallback | None
    ) -> None:
        node = shutil.which(self._node)
        if node is None:
            raise RenderEngineError(f"No se encuentra el binario de Node.js ('{self._node}').")
        if not self._script.is_file():
            raise RenderEngineError(f"No existe el script del worker de render: {self._script}")

        cmd = [node, str(self._script), "--payload", str(payload_path), "--out", str(frames_dir)]
        logger.info("Lanzando worker de render: %s", " ".join(cmd))
        try:
            result = subprocess.run(  # noqa: S603 - lista de argumentos, sin shell
                cmd, capture_output=True, text=True, timeout=self._timeout, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderEngineError(f"El worker de render superó el tiempo límite ({self._timeout} s).") from exc
        except OSError as exc:
            raise RenderEngineError("No se pudo ejecutar el worker de render.", detail=str(exc)) from exc
        if result.returncode != 0:
            raise RenderEngineError(
                f"El worker de render terminó con código {result.returncode}.", detail=result.stderr[-2000:]
            )
        if on_progress:
            on_progress(payload.frame_count, payload.frame_count)


# --------------------------------------------------------------------------- #
# Motor simulado
# --------------------------------------------------------------------------- #
class SimulatedEngine:
    """Sustituto del motor 3D: mapa cenital 2D + perfil altimétrico animados.

    Produce exactamente los mismos artefactos que el motor real (PNG
    secuenciales con la resolución pedida), lo que permite validar de extremo
    a extremo la ingesta, la orquestación y el ensamblaje con FFmpeg.
    """

    name = "simulated"

    _BG = (17, 21, 28)
    _TRACK = (70, 78, 92)
    _DONE = (255, 107, 53)
    _TEXT = (235, 238, 242)

    def render(
        self, payload: RenderPayload, payload_path: Path, frames_dir: Path, on_progress: ProgressCallback | None
    ) -> None:
        w, h = payload.width, payload.height
        frames = payload.frames
        n = len(frames)
        xs = np.fromiter((f.x for f in frames), dtype=np.float64, count=n)
        ys = np.fromiter((f.y for f in frames), dtype=np.float64, count=n)
        zs = np.fromiter((f.z for f in frames), dtype=np.float64, count=n)
        ds = np.fromiter((f.dist for f in frames), dtype=np.float64, count=n)

        # Área del mapa (78 % superior) y del perfil (franja inferior)
        margin = h * 0.08
        map_h = h * 0.74
        span_x, span_y = max(np.ptp(xs), 1.0), max(np.ptp(ys), 1.0)
        scale = min((w - 2 * margin) / span_x, (map_h - 2 * margin) / span_y)
        px = (w - span_x * scale) / 2 + (xs - xs.min()) * scale
        py = (map_h + span_y * scale) / 2 - (ys - ys.min()) * scale  # el eje Y de imagen crece hacia abajo

        prof_top, prof_bottom = h * 0.80, h * 0.95
        z_span = max(np.ptp(zs), 1.0)
        qx = margin + ds / max(ds[-1], 1.0) * (w - 2 * margin)
        qy = prof_bottom - (zs - zs.min()) / z_span * (prof_bottom - prof_top)

        line_w = max(h // 200, 2)
        radius = max(h // 90, 4)
        font = ImageFont.load_default(size=max(h // 36, 12))
        track_pts = list(zip(px.tolist(), py.tolist()))
        prof_pts = list(zip(qx.tolist(), qy.tolist()))

        canvas = Image.new("RGB", (w, h), self._BG)
        base = ImageDraw.Draw(canvas)
        base.line(track_pts, fill=self._TRACK, width=line_w, joint="curve")
        base.polygon([(qx[0], prof_bottom), *prof_pts, (qx[-1], prof_bottom)], fill=(38, 44, 56))
        base.line(prof_pts, fill=self._TRACK, width=max(line_w // 2, 1))

        try:
            for i, frame in enumerate(frames):
                if i > 0:  # el tramo recorrido se acumula sobre el lienzo persistente
                    base.line([track_pts[i - 1], track_pts[i]], fill=self._DONE, width=line_w + 1)
                    base.line([prof_pts[i - 1], prof_pts[i]], fill=self._DONE, width=line_w)
                img = canvas.copy()
                d = ImageDraw.Draw(img)
                cx, cy = track_pts[i]
                d.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=self._TEXT, outline=self._DONE, width=3)
                d.line([(prof_pts[i][0], prof_top), (prof_pts[i][0], prof_bottom)], fill=self._TEXT, width=1)
                secs = int(frame.elapsed)
                hud = (
                    f"{frame.dist / 1000:6.2f} km   {frame.ele:5.0f} m   "
                    f"{secs // 3600:d}:{secs % 3600 // 60:02d}:{secs % 60:02d}   [{payload.camera.style.value}]"
                )
                d.text((margin, margin * 0.4), hud, fill=self._TEXT, font=font)
                img.save(frames_dir / FRAME_NAME.format(i), format="PNG", compress_level=1)
                if on_progress and (i % 15 == 0 or i == n - 1):
                    on_progress(i + 1, n)
        except OSError as exc:
            raise RenderEngineError("Error de E/S al escribir los fotogramas.", detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Orquestador
# --------------------------------------------------------------------------- #
def _camera_config(style: CameraStyle, track: ProcessedTrack) -> CameraConfig:
    """Parámetros de cámara según estilo, escalados a la extensión del recorrido."""
    extent = float(max(np.ptp(track.x), np.ptp(track.y), 200.0))
    if style is CameraStyle.OVERHEAD:
        return CameraConfig(style=style, distance_m=round(extent * 0.6, 1), pitch_deg=89.0)
    if style is CameraStyle.ORBIT:
        return CameraConfig(style=style, distance_m=round(max(extent * 0.35, 800.0), 1), pitch_deg=35.0, orbit_deg_per_s=6.0)
    return CameraConfig(style=style, distance_m=round(float(np.clip(extent * 0.3, 700.0, 6000.0)), 1), pitch_deg=33.0)


def _camera_positions(cfg: CameraConfig, track: ProcessedTrack) -> np.ndarray:
    """Posición (x, y, z) de la cámara en cada fotograma, mirando al punto activo."""
    if cfg.style is CameraStyle.FOLLOW:
        azimuth = np.deg2rad(track.heading_deg)  # dirección hacia la que mira la cámara
    elif cfg.style is CameraStyle.ORBIT:
        azimuth = np.deg2rad(track.heading_deg[0] + cfg.orbit_deg_per_s * track.video_t)
    else:
        azimuth = np.zeros(track.frame_count)  # norte arriba
    pitch = np.deg2rad(cfg.pitch_deg)
    ground = cfg.distance_m * np.cos(pitch)
    return np.column_stack(
        (
            track.x - ground * np.sin(azimuth),
            track.y - ground * np.cos(azimuth),
            track.ele * cfg.vertical_exaggeration + cfg.distance_m * np.sin(pitch),
        )
    )


def create_engine(settings: Settings) -> RenderEngine:
    """Fábrica de motores según ``settings.render_engine``."""
    if settings.render_engine == "node":
        return NodeThreeJsEngine(settings.node_binary, settings.node_worker_script, settings.render_timeout_s)
    if settings.render_engine == "terrain3d":
        from .terrain_engine import Terrain3DEngine  # import perezoso: moderngl solo si se usa

        return Terrain3DEngine(settings)
    if settings.render_engine == "simulated":
        return SimulatedEngine()
    raise RenderEngineError(f"Motor de render desconocido: '{settings.render_engine}'.")


class RendererOrchestrator:
    """Coordina payload → motor gráfico → verificación de fotogramas."""

    def __init__(self, engine: RenderEngine | None = None, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._engine = engine or create_engine(self._settings)

    def build_payload(self, job_id: str, track: ProcessedTrack, options: RenderOptions) -> RenderPayload:
        """Construye el contrato JSON que consume el motor gráfico."""
        width, height = options.resolution.size
        camera = _camera_config(options.camera_style, track)
        cam = _camera_positions(camera, track)
        frames = [
            FramePoint(
                i=i,
                t=round(float(track.video_t[i]), 4),
                elapsed=round(float(track.elapsed_s[i]), 2),
                lat=round(float(track.lat[i]), 7),
                lon=round(float(track.lon[i]), 7),
                ele=round(float(track.ele[i]), 2),
                x=round(float(track.x[i]), 2),
                y=round(float(track.y[i]), 2),
                z=round(float(track.ele[i]), 2),
                dist=round(float(track.distance_m[i]), 2),
                heading=round(float(track.heading_deg[i]), 2),
                cam=(round(float(cam[i, 0]), 2), round(float(cam[i, 1]), 2), round(float(cam[i, 2]), 2)),
            )
            for i in range(track.frame_count)
        ]
        return RenderPayload(
            job_id=job_id,
            width=width,
            height=height,
            fps=track.fps,
            frame_count=track.frame_count,
            duration_s=track.video_duration_s,
            origin={"lat": track.origin_lat, "lon": track.origin_lon},
            bounds={
                "min_lat": float(track.lat.min()),
                "max_lat": float(track.lat.max()),
                "min_lon": float(track.lon.min()),
                "max_lon": float(track.lon.max()),
                "min_ele": float(track.ele.min()),
                "max_ele": float(track.ele.max()),
            },
            camera=camera,
            metrics=track.metrics,
            frames=frames,
        )

    def render_frames(
        self,
        payload: RenderPayload,
        workdir: Path,
        on_progress: ProgressCallback | None = None,
        *,
        verify: bool = True,
    ) -> Path:
        """Escribe el payload, ejecuta el motor y valida la secuencia resultante.

        Args:
            payload: contrato de render.
            workdir: directorio temporal exclusivo del trabajo.
            on_progress: callback ``(hechos, total)``; cuando se invoca, los fotogramas
                ``[0, hechos)`` ya están completos en disco (permite consumirlos en streaming).
            verify: comprobar al final que la secuencia está completa. Debe ser ``False``
                si el consumidor va borrando los fotogramas a medida que los codifica.

        Returns:
            Directorio que contiene ``frame_0000.png … frame_{N-1}.png``.

        Raises:
            RenderEngineError: fallo del motor o secuencia incompleta/corrupta.
        """
        frames_dir = workdir / "frames"
        payload_path = workdir / "payload.json"
        try:
            frames_dir.mkdir(parents=True, exist_ok=False)
            payload_path.write_text(payload.model_dump_json(), encoding="utf-8")
        except OSError as exc:
            raise RenderEngineError("No se pudo preparar el directorio de trabajo.", detail=str(exc)) from exc

        logger.info("Job %s: renderizando %d fotogramas con motor '%s'", payload.job_id, payload.frame_count, self._engine.name)
        try:
            self._engine.render(payload, payload_path, frames_dir, on_progress)
        except GPXRenderError:  # incluye errores del consumidor de fotogramas (FFmpeg)
            raise
        except Exception as exc:  # noqa: BLE001 - un motor de terceros no debe tumbar el worker
            raise RenderEngineError("Fallo inesperado del motor de render.", detail=repr(exc)) from exc

        if verify:
            self.verify_frames(frames_dir, payload.frame_count)
        return frames_dir

    @staticmethod
    def verify_frames(frames_dir: Path, expected: int) -> None:
        """Comprueba que la secuencia es completa, contigua y son PNG reales."""
        indices = sorted(
            int(m.group(1)) for p in frames_dir.iterdir() if p.is_file() and (m := FRAME_REGEX.match(p.name))
        )
        if len(indices) != expected or indices != list(range(expected)):
            missing = sorted(set(range(expected)) - set(indices))[:5]
            raise RenderEngineError(
                f"Secuencia de fotogramas incompleta: se esperaban {expected} y hay {len(indices)}"
                + (f" (faltan, p. ej., {missing})." if missing else ".")
            )
        for idx in (0, expected - 1):  # comprobación barata de integridad en los extremos
            path = frames_dir / FRAME_NAME.format(idx)
            with path.open("rb") as fh:
                if fh.read(8) != PNG_SIGNATURE:
                    raise RenderEngineError(f"El fotograma {path.name} no es un PNG válido.")

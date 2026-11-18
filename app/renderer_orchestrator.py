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
from .deadline import check_deadline
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
    _WHITE_ROUTE = (255, 255, 255)

    @staticmethod
    def _ease(t: float) -> float:
        # easeInOutSine suave
        import math as _m

        return 0.5 * (1 - _m.cos(_m.pi * t))

    def _sim_base(self, payload: RenderPayload):
        frames = payload.frames
        n = len(frames)
        xs = np.fromiter((f.x for f in frames), dtype=np.float64, count=n)
        ys = np.fromiter((f.y for f in frames), dtype=np.float64, count=n)
        zs = np.fromiter((f.z for f in frames), dtype=np.float64, count=n)
        ds = np.fromiter((f.dist for f in frames), dtype=np.float64, count=n)
        return xs, ys, zs, ds

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
                check_deadline()  # plazo cooperativo del trabajo en curso
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

    # --- Cinemática ----------------------------------------------------
    def _draw_stats_panel(self, img: Image.Image, payload: RenderPayload) -> None:
        """Panel semitransparente con métricas finales (outro)."""
        w, h = img.size
        m = payload.metrics
        # Formateos
        dist_km = m.total_distance_m / 1000.0
        dur_s = int(m.duration_s)
        hms = f"{dur_s // 3600}:{(dur_s % 3600)//60:02d}:{dur_s % 60:02d}"
        if m.duration_is_estimated:
            hms += " *"
        gain = f"{m.elevation_gain_m:.0f} m"
        loss = f"{m.elevation_loss_m:.0f} m"
        elev = f"{m.min_elevation_m:.0f} – {m.max_elevation_m:.0f} m" if m.has_elevation else "—"
        avg = f"{m.avg_speed_kmh:.1f} km/h"
        # Panel
        panel_w = int(w * 0.72)
        panel_h = int(h * 0.32)
        x0 = (w - panel_w) // 2
        y0 = int(h * 0.60)
        x1 = x0 + panel_w
        y1 = y0 + panel_h
        d = ImageDraw.Draw(img, "RGBA")
        radius = h // 50
        d.rounded_rectangle((x0, y0, x1, y1), radius, fill=(0, 0, 0, 170), outline=(255, 92, 30, 200), width=2)
        # Fuentes
        f_title = ImageFont.load_default(size=max(h // 28, 14))
        f_small = ImageFont.load_default(size=max(h // 40, 11))
        title = payload.metrics  # keep for mypy
        # Título
        t = "Resumen de ruta"
        # usar textbbox para centrar
        bbox = d.textbbox((0, 0), t, font=f_title)
        tw = bbox[2] - bbox[0]
        d.text(((w - tw) // 2, y0 + panel_h * 0.08), t, font=f_title, fill=(255, 255, 255, 255))
        # Línea separadora
        d.line([(x0 + 20, y0 + panel_h * 0.24), (x1 - 20, y0 + panel_h * 0.24)], fill=(255, 255, 255, 60), width=1)
        # Grid 3x2
        cols = [f"{dist_km:.2f} km", hms, gain, loss, elev, avg]
        labels = ["Distancia", "Tiempo", "Desnivel +", "Desnivel -", "Altitud", "Vel. media"]
        col_w = panel_w // 3
        row_h = int(panel_h * 0.30)
        for idx, (lab, val) in enumerate(zip(labels, cols)):
            col = idx % 3
            row = idx // 3
            cx = x0 + col * col_w + col_w // 2
            cy = y0 + panel_h * 0.32 + row * row_h
            # label
            lb = d.textbbox((0, 0), lab, font=f_small)
            lw = lb[2] - lb[0]
            d.text((cx - lw // 2, cy), lab, font=f_small, fill=(200, 200, 200, 255))
            # value
            vb = d.textbbox((0, 0), val, font=f_title)
            vw = vb[2] - vb[0]
            d.text((cx - vw // 2, cy + h * 0.04), val, font=f_title, fill=(255, 255, 255, 255))

    def render_intro(self, payload: RenderPayload, frames_dir: Path, on_progress: ProgressCallback | None) -> None:
        """Intro simulada: overview lejano → zoom al punto de salida ajustado al camera_style inicial."""
        w, h = payload.width, payload.height
        fps = payload.fps
        xs, ys, _, _ = self._sim_base(payload)
        frames_dir.mkdir(parents=True, exist_ok=True)
        margin = h * 0.08
        map_h = h * 0.74
        span_x, span_y = max(np.ptp(xs), 1.0), max(np.ptp(ys), 1.0)
        scale_overview = min((w - 2 * margin) / span_x, (map_h - 2 * margin) / span_y)
        # overview lejano: 55% del overview → se ve ~1.8× más terreno alrededor
        scale_far = scale_overview * 0.55
        center_ov_x = float((xs.min() + xs.max()) / 2)
        center_ov_y = float((ys.min() + ys.max()) / 2)
        center_start_x = float(xs[0])
        center_start_y = float(ys[0])
        # Escala de cierre ajustada al estilo: overhead no hace zoom (evita salto con tracking estático)
        # follow/orbit cierran a 3.2× overview; overhead se queda en overview (sin salto)
        style = payload.camera.style
        if style.value == "overhead":
            scale_close = scale_overview  # sin zoom, solo centrado
            close_cx, close_cy = center_ov_x, center_ov_y
            # pero para que se note zoom, usamos ligero acercamiento 1.3× manteniendo centro
            scale_close = scale_overview * 1.25
        elif style.value == "orbit":
            scale_close = scale_overview * 3.0
            close_cx, close_cy = center_start_x, center_start_y
        else:  # follow
            scale_close = scale_overview * 3.4
            close_cx, close_cy = center_start_x, center_start_y
        # Para overhead el centro final es overview (no salto), para follow/orbit es el punto de salida
        if style.value == "overhead":
            close_cx, close_cy = center_ov_x, center_ov_y
        from .config import get_settings as _gs
        s = _gs()
        n = int(round(s.cinematic_intro_s * fps))
        hold = int(round(1.0 * fps))
        zoom_n = max(n - hold, 1)
        line_w = max(h // 200, 2)
        radius = max(h // 70, 6)
        font = ImageFont.load_default(size=max(h // 36, 12))
        for i in range(n):
            check_deadline()
            if i < hold:
                t = 0.0
            else:
                t = (i - hold) / max(zoom_n - 1, 1)
                t = self._ease(t)
            # Interpolación híbrida: centro y escala con ease
            cx_w = center_ov_x + (close_cx - center_ov_x) * t
            cy_w = center_ov_y + (close_cy - center_ov_y) * t
            sc = scale_far + (scale_close - scale_far) * t
            px = w / 2 + (xs - cx_w) * sc
            py = map_h / 2 - (ys - cy_w) * sc
            track_pts = list(zip(px.tolist(), py.tolist()))
            img = Image.new("RGB", (w, h), self._BG)
            d = ImageDraw.Draw(img)
            d.line(track_pts, fill=self._WHITE_ROUTE, width=line_w + 1, joint="curve")
            # marcador inicio: en el punto de salida (centro final cuando t=1 queda centrado)
            sx = w / 2 + (center_start_x - cx_w) * sc
            sy = map_h / 2 - (center_start_y - cy_w) * sc
            r = radius + int(2 * t) if t > 0 else radius
            d.ellipse((sx - r, sy - r, sx + r, sy + r), fill=(255, 255, 255), outline=(255, 92, 30), width=3)
            # HUD estilo
            label = {"follow": "Inicio · Follow", "orbit": "Inicio · Orbit", "overhead": "Inicio · Cenital"}.get(style.value, "Inicio")
            d.text((margin, margin * 0.4), label, fill=self._TEXT, font=font)
            img.save(frames_dir / FRAME_NAME.format(i), format="PNG", compress_level=1)
            if on_progress and (i % 15 == 0 or i == n - 1):
                on_progress(i + 1, n)

    def render_outro(self, payload: RenderPayload, frames_dir: Path, on_progress: ProgressCallback | None) -> None:
        """Outro simulado: comienza donde acabó el tracking → zoom-out a overview lejano."""
        w, h = payload.width, payload.height
        fps = payload.fps
        xs, ys, _, _ = self._sim_base(payload)
        frames_dir.mkdir(parents=True, exist_ok=True)
        margin = h * 0.08
        map_h = h * 0.74
        span_x, span_y = max(np.ptp(xs), 1.0), max(np.ptp(ys), 1.0)
        scale_overview = min((w - 2 * margin) / span_x, (map_h - 2 * margin) / span_y)
        scale_far = scale_overview * 0.55
        center_ov_x = float((xs.min() + xs.max()) / 2)
        center_ov_y = float((ys.min() + ys.max()) / 2)
        center_end_x = float(xs[-1])
        center_end_y = float(ys[-1])
        style = payload.camera.style
        if style.value == "overhead":
            scale_close = scale_overview * 1.25
            close_cx, close_cy = center_ov_x, center_ov_y
        elif style.value == "orbit":
            scale_close = scale_overview * 3.0
            close_cx, close_cy = center_end_x, center_end_y
        else:
            scale_close = scale_overview * 3.4
            close_cx, close_cy = center_end_x, center_end_y
        if style.value == "overhead":
            close_cx, close_cy = center_ov_x, center_ov_y
        from .config import get_settings as _gs
        s = _gs()
        n = int(round(s.cinematic_outro_s * fps))
        zoom_n = int(round(2.0 * fps))
        if zoom_n > n - 15:
            zoom_n = max(n - 30, 1)
        stats_start = n - int(round(1.5 * fps))
        line_w = max(h // 200, 2)
        radius = max(h // 70, 6)
        font = ImageFont.load_default(size=max(h // 36, 12))
        for i in range(n):
            check_deadline()
            if i < zoom_n:
                t = i / max(zoom_n - 1, 1)
                t = self._ease(t)
                # Outro: de close (igual que último frame tracking) a far overview lejano
                cx_w = close_cx + (center_ov_x - close_cx) * t
                cy_w = close_cy + (center_ov_y - close_cy) * t
                sc = scale_close + (scale_far - scale_close) * t
            else:
                cx_w, cy_w, sc = center_ov_x, center_ov_y, scale_far
            px = w / 2 + (xs - cx_w) * sc
            py = map_h / 2 - (ys - cy_w) * sc
            track_pts = list(zip(px.tolist(), py.tolist()))
            img = Image.new("RGB", (w, h), self._BG)
            d = ImageDraw.Draw(img)
            d.line(track_pts, fill=self._DONE, width=line_w + 1, joint="curve")
            if i < zoom_n:
                ex = w / 2 + (center_end_x - cx_w) * sc
                ey = map_h / 2 - (center_end_y - cy_w) * sc
                d.ellipse((ex - radius, ey - radius, ex + radius, ey + radius), fill=(255, 92, 30), outline=(255, 255, 255), width=3)
                label = {"follow": "Meta · Follow", "orbit": "Meta · Orbit", "overhead": "Meta · Cenital"}.get(style.value, "Meta")
                d.text((margin, margin * 0.4), label, fill=self._TEXT, font=font)
            if i >= stats_start:
                self._draw_stats_panel(img, payload)
            elif i >= zoom_n:
                d.text((margin, margin * 0.4), f"{payload.metrics.total_distance_m/1000:.2f} km", fill=self._TEXT, font=font)
            img.save(frames_dir / FRAME_NAME.format(i), format="PNG", compress_level=1)
            if on_progress and (i % 15 == 0 or i == n - 1):
                on_progress(i + 1, n)


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


def _smooth_heading_deg(heading_deg: np.ndarray, sigma: float) -> np.ndarray:
    """Suaviza ángulos circulares vía sin/cos para evitar el salto 359→0."""
    from scipy.ndimage import gaussian_filter1d as _g

    rad = np.deg2rad(heading_deg)
    return np.rad2deg(np.arctan2(_g(np.sin(rad), sigma=sigma, mode="nearest"), _g(np.cos(rad), sigma=sigma, mode="nearest"))) % 360.0


def _camera_positions(cfg: CameraConfig, track: ProcessedTrack) -> np.ndarray:
    """Posición (x, y, z) de la cámara en cada fotograma, mirando al punto activo."""
    # Para FOLLOW el rumbo crudo puede dar bandazos en horquillas; se suaviza el yaw ~1.8s
    if cfg.style is CameraStyle.FOLLOW:
        # track.heading_deg ya viene suavizado 0.75s en gpx_processor; añadimos ~1.8s extra aquí
        hdg = _smooth_heading_deg(track.heading_deg, sigma=track.fps * 1.8)
        azimuth = np.deg2rad(hdg)
    elif cfg.style is CameraStyle.ORBIT:
        azimuth = np.deg2rad(track.heading_deg[0] + cfg.orbit_deg_per_s * track.video_t)
    else:
        azimuth = np.zeros(track.frame_count)  # norte arriba (OVERHEAD): sin yaw, paneo cenital puro
    pitch = np.deg2rad(cfg.pitch_deg)
    ground = cfg.distance_m * np.cos(pitch)
    eyes = np.column_stack(
        (
            track.x - ground * np.sin(azimuth),
            track.y - ground * np.cos(azimuth),
            track.ele * cfg.vertical_exaggeration + cfg.distance_m * np.sin(pitch),
        )
    )
    # Suavizado extra de la trayectoria de ojos: más fuerte en OVERHEAD (toda la sensación de movimiento es traslación)
    if cfg.style is CameraStyle.OVERHEAD:
        from scipy.ndimage import gaussian_filter1d as _g

        eyes = _g(eyes, sigma=track.fps * 2.0, axis=0, mode="nearest")
    return eyes


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

    @property
    def engine(self) -> RenderEngine:  # type: ignore[return]
        return self._engine

    def render_intro(
        self, payload: RenderPayload, frames_dir: Path, on_progress: ProgressCallback | None = None
    ) -> Path:
        """Intro cinemático: ruta completa en blanco + zoom al inicio."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        # B-4: payload.json solo para motor node (el resto no lo lee)
        if self._settings.render_engine == "node":
            try:
                (frames_dir.parent / "payload.json").write_text(payload.model_dump_json(), encoding="utf-8")
            except OSError:
                pass
        logger.info("Job %s: intro %s (%d fotogramas)", payload.job_id, self._engine.name, int(self._settings.cinematic_intro_s * payload.fps))
        try:
            if hasattr(self._engine, "render_intro"):
                self._engine.render_intro(payload, frames_dir, on_progress)  # type: ignore[attr-defined]
            else:
                # fallback: genera fotogramas vacíos si el motor no soporta cinemática
                raise RenderEngineError("Motor sin soporte de intro.")
        except GPXRenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderEngineError("Fallo intro cinemático.", detail=repr(exc)) from exc
        return frames_dir

    def render_outro(
        self, payload: RenderPayload, frames_dir: Path, on_progress: ProgressCallback | None = None
    ) -> Path:
        """Outro cinemático: zoom desde el final + stats."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Job %s: outro %s (%d fotogramas)", payload.job_id, self._engine.name, int(self._settings.cinematic_outro_s * payload.fps))
        try:
            if hasattr(self._engine, "render_outro"):
                self._engine.render_outro(payload, frames_dir, on_progress)  # type: ignore[attr-defined]
            else:
                raise RenderEngineError("Motor sin soporte de outro.")
        except GPXRenderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RenderEngineError("Fallo outro cinemático.", detail=repr(exc)) from exc
        return frames_dir

    def build_payload(self, job_id: str, track: ProcessedTrack, options: RenderOptions) -> RenderPayload:
        """Construye el contrato JSON que consume el motor gráfico."""
        # Soporta orientación vertical (9:16) para redes sociales
        try:
            width, height = options.dimensions()  # type: ignore[attr-defined]
        except AttributeError:
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
            # B-4: solo serializar payload.json para engine node (~2MB saved por trabajo)
            if self._settings.render_engine == "node":
                payload_path.write_text(payload.model_dump_json(), encoding="utf-8")
            else:
                payload_path.write_text("{}", encoding="utf-8")
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

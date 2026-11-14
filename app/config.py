"""Configuración centralizada del servicio (sobrescribible por variables de entorno)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    """Parámetros de funcionamiento. Todas las variables de entorno usan el prefijo ``GPX_``."""

    # --- Ingesta -----------------------------------------------------------
    max_upload_bytes: int = field(default_factory=lambda: _env_int("GPX_MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
    max_track_points: int = field(default_factory=lambda: _env_int("GPX_MAX_TRACK_POINTS", 500_000))

    # --- Suavizado ---------------------------------------------------------
    #: Ventana (nº de muestras, impar) del filtro de mediana/Hampel anti-saltos.
    outlier_window: int = 7
    #: Desviación mínima (m) respecto a la mediana local para considerar un punto como salto GPS.
    outlier_min_deviation_m: float = 25.0
    #: Ventana (impar) del filtro Savitzky-Golay aplicado a la posición horizontal.
    position_smooth_window: int = 11
    #: Ventana (impar) del filtro Savitzky-Golay aplicado a la altitud (más ruidosa).
    elevation_smooth_window: int = 21
    #: Histéresis (m) para acumular desnivel: ignora oscilaciones menores.
    elevation_gain_threshold_m: float = 3.0
    #: Por debajo de esta velocidad (m/s) el tramo se considera parado.
    moving_speed_threshold_ms: float = 0.5

    # --- Animación ---------------------------------------------------------
    fps: int = 30  # Requisito: 30 FPS
    seconds_per_km: float = field(default_factory=lambda: _env_float("GPX_SECONDS_PER_KM", 4.0))
    min_video_seconds: float = field(default_factory=lambda: _env_float("GPX_MIN_VIDEO_SECONDS", 20.0))
    max_video_seconds: float = field(default_factory=lambda: _env_float("GPX_MAX_VIDEO_SECONDS", 90.0))

    # --- Motor de render ---------------------------------------------------
    #: ``terrain3d`` (relieve + satélite con OpenGL headless), ``simulated`` (2D, sin red ni OpenGL)
    #: o ``node`` (worker Three.js externo).
    render_engine: str = field(default_factory=lambda: os.environ.get("GPX_RENDER_ENGINE", "terrain3d"))
    node_binary: str = field(default_factory=lambda: os.environ.get("GPX_NODE_BINARY", "node"))
    node_worker_script: str = field(
        default_factory=lambda: os.environ.get("GPX_NODE_WORKER_SCRIPT", "render_worker/worker.js")
    )
    render_timeout_s: int = field(default_factory=lambda: _env_int("GPX_RENDER_TIMEOUT_S", 3600))
    max_concurrent_renders: int = field(default_factory=lambda: _env_int("GPX_MAX_CONCURRENT_RENDERS", 2))

    # --- Terreno 3D (motor ``terrain3d``) ------------------------------------
    #: Teselas de elevación en codificación *terrarium* (plantilla con {z}/{x}/{y}).
    dem_url: str = field(
        default_factory=lambda: os.environ.get(
            "GPX_DEM_URL", "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
        )
    )
    #: Teselas de imagen satélite XYZ (plantilla con {z}/{x}/{y}).
    imagery_url: str = field(
        default_factory=lambda: os.environ.get(
            "GPX_IMAGERY_URL",
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        )
    )
    imagery_attribution: str = field(
        default_factory=lambda: os.environ.get(
            "GPX_IMAGERY_ATTRIBUTION", "Imagery © Esri, Maxar, Earthstar Geographics · Terrain: Mapzen/AWS"
        )
    )
    imagery_max_px: int = field(default_factory=lambda: _env_int("GPX_IMAGERY_MAX_PX", 6144))
    imagery_max_zoom: int = 17
    dem_max_px: int = 1024
    dem_max_zoom: int = 14
    #: Vértices por lado (máx.) de la malla del terreno.
    terrain_mesh_max: int = field(default_factory=lambda: _env_int("GPX_TERRAIN_MESH_MAX", 480))
    terrain_margin_ratio: float = 0.4
    terrain_min_margin_m: float = 1500.0
    max_tiles_per_layer: int = 700
    tile_download_workers: int = 12
    tile_timeout_s: float = 20.0
    tile_cache_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("GPX_TILE_CACHE_DIR", str(Path(tempfile.gettempdir()) / "gpx_render_tile_cache"))
        )
    )

    # --- FFmpeg ------------------------------------------------------------
    ffmpeg_binary: str = field(default_factory=lambda: os.environ.get("GPX_FFMPEG_BINARY", "ffmpeg"))
    ffmpeg_crf: int = field(default_factory=lambda: _env_int("GPX_FFMPEG_CRF", 20))
    ffmpeg_preset: str = field(default_factory=lambda: os.environ.get("GPX_FFMPEG_PRESET", "medium"))
    ffmpeg_timeout_s: int = field(default_factory=lambda: _env_int("GPX_FFMPEG_TIMEOUT_S", 1800))

    # --- Almacenamiento ----------------------------------------------------
    output_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("GPX_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "gpx_render_output"))
        )
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la configuración (singleton perezoso)."""
    return Settings()

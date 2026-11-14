"""Modelos Pydantic: opciones de entrada, métricas, payload de render y estado de trabajos."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Resolution(str, Enum):
    """Resoluciones de salida soportadas."""

    P720 = "720p"
    P1080 = "1080p"

    @property
    def size(self) -> tuple[int, int]:
        """Dimensiones ``(ancho, alto)`` en píxeles."""
        return (1280, 720) if self is Resolution.P720 else (1920, 1080)


class CameraStyle(str, Enum):
    """Estilos de cámara disponibles para la animación."""

    FOLLOW = "follow"  # Persecución: detrás del punto activo, orientada según el rumbo
    ORBIT = "orbit"  # Órbita lenta alrededor del punto activo
    OVERHEAD = "overhead"  # Vista cenital con el norte arriba


class RenderOptions(BaseModel):
    """Parámetros opcionales de la animación."""

    model_config = ConfigDict(frozen=True)

    animation_speed: float = Field(
        default=1.0, ge=0.5, le=4.0, description="Multiplicador de velocidad (2.0 = vídeo la mitad de largo)."
    )
    resolution: Resolution = Resolution.P1080
    camera_style: CameraStyle = CameraStyle.FOLLOW


class TrackMetrics(BaseModel):
    """Métricas globales del track."""

    point_count: int = Field(description="Puntos válidos tras la limpieza.")
    total_distance_m: float
    elevation_gain_m: float
    elevation_loss_m: float
    min_elevation_m: float
    max_elevation_m: float
    duration_s: float = Field(description="Duración total (real si hay marcas de tiempo; estimada si no).")
    moving_time_s: float
    duration_is_estimated: bool
    avg_speed_kmh: float = Field(description="Distancia / duración total.")
    avg_moving_speed_kmh: float = Field(description="Distancia / tiempo en movimiento.")
    has_elevation: bool
    has_timestamps: bool
    outliers_corrected: int = Field(description="Saltos GPS detectados y corregidos.")
    start_time: datetime | None = None


class FramePoint(BaseModel):
    """Estado de la escena en un fotograma concreto."""

    i: int = Field(description="Índice de fotograma (base 0).")
    t: float = Field(description="Tiempo relativo dentro del vídeo (s).")
    elapsed: float = Field(description="Tiempo transcurrido de la actividad real (s).")
    lat: float
    lon: float
    ele: float
    x: float = Field(description="Este local (m) respecto al origen.")
    y: float = Field(description="Norte local (m) respecto al origen.")
    z: float = Field(description="Altitud (m).")
    dist: float = Field(description="Distancia acumulada (m).")
    heading: float = Field(description="Rumbo suavizado (grados, 0 = norte, sentido horario).")
    cam: tuple[float, float, float] = Field(description="Posición de cámara sugerida (x, y, z) en metros locales.")


class CameraConfig(BaseModel):
    """Parámetros de cámara que el motor gráfico debe aplicar."""

    style: CameraStyle
    distance_m: float
    pitch_deg: float
    fov_deg: float = 45.0
    orbit_deg_per_s: float = 0.0
    vertical_exaggeration: float = 1.5


class RenderPayload(BaseModel):
    """Contrato JSON entre el backend y el motor gráfico headless."""

    schema_version: str = "1.0"
    job_id: str
    width: int
    height: int
    fps: int
    frame_count: int
    duration_s: float
    frame_pattern: str = Field(default="frame_%04d.png", description="Nombre esperado de cada fotograma.")
    origin: dict[str, float] = Field(description="Origen (lat, lon) de la proyección local en metros.")
    bounds: dict[str, float]
    camera: CameraConfig
    metrics: TrackMetrics
    frames: list[FramePoint]


class JobStatus(str, Enum):
    """Fases del ciclo de vida de un trabajo de render."""

    QUEUED = "queued"
    RENDERING = "rendering"
    COMPILING = "compiling"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    """Estado público de un trabajo."""

    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=1.0)
    options: RenderOptions
    metrics: TrackMetrics
    frame_count: int
    video_duration_s: float
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    status_url: str
    video_url: str | None = None


class ErrorResponse(BaseModel):
    """Cuerpo estándar de error."""

    error: str
    message: str

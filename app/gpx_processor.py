"""Parsing, limpieza, métricas y suavizado de tracks GPX.

Flujo de :func:`process_gpx`::

    bytes ──parse_gpx──▶ GPX ──extract_points──▶ RawTrack
          ──proyección local (m)──▶ corrección de saltos (Hampel)
          ──métricas──▶ Savitzky-Golay ──▶ remuestreo uniforme por distancia
          ──▶ ProcessedTrack (un punto por fotograma)

Todo el filtrado se hace en un plano local en metros (Este/Norte), no en
grados: así los umbrales tienen sentido físico y el suavizado es isótropo.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime

import gpxpy
import gpxpy.gpx
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter

from .config import Settings, get_settings
from .exceptions import InvalidGPXError
from .schemas import RenderOptions, TrackMetrics

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

EARTH_RADIUS_M = 6_371_008.8
_FORBIDDEN_XML = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Estructuras de datos
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawTrack:
    """Puntos del track tal cual vienen en el GPX (ya validados)."""

    lat: FloatArray
    lon: FloatArray  # "desenrollada": continua aunque se cruce el antimeridiano
    ele: FloatArray
    time_s: FloatArray | None  # segundos desde el primer punto; None si no hay tiempos fiables
    has_elevation: bool
    start_time: datetime | None

    def __len__(self) -> int:
        return int(self.lat.size)


@dataclass(frozen=True)
class ProcessedTrack:
    """Trayectoria suavizada y remuestreada: un elemento por fotograma de vídeo."""

    metrics: TrackMetrics
    fps: int
    video_duration_s: float
    origin_lat: float
    origin_lon: float
    lat: FloatArray
    lon: FloatArray
    ele: FloatArray
    x: FloatArray  # Este local (m)
    y: FloatArray  # Norte local (m)
    distance_m: FloatArray
    video_t: FloatArray  # marca de tiempo relativa dentro del vídeo (s)
    elapsed_s: FloatArray  # tiempo real de actividad (s)
    heading_deg: FloatArray

    @property
    def frame_count(self) -> int:
        return int(self.lat.size)

    def to_points(self) -> list[dict[str, float]]:
        """Serializa la trayectoria como lista JSON normalizada de puntos."""
        return [
            {
                "t": round(float(self.video_t[i]), 4),
                "elapsed": round(float(self.elapsed_s[i]), 2),
                "lat": round(float(self.lat[i]), 7),
                "lon": round(float(self.lon[i]), 7),
                "ele": round(float(self.ele[i]), 2),
                "dist": round(float(self.distance_m[i]), 2),
                "heading": round(float(self.heading_deg[i]), 2),
            }
            for i in range(self.frame_count)
        ]


# --------------------------------------------------------------------------- #
# Parsing y validación
# --------------------------------------------------------------------------- #
def _decode(data: bytes) -> str:
    """Decodifica el XML respetando BOM; recurre a latin-1 como último recurso."""
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeError:
            continue
    return data.decode("latin-1")


def parse_gpx(data: bytes) -> gpxpy.gpx.GPX:
    """Valida y parsea el contenido de un archivo GPX.

    Raises:
        InvalidGPXError: contenido vacío, XML mal formado, raíz distinta de
            ``<gpx>`` o presencia de DTD/entidades (vector de ataques XXE y
            "billion laughs", innecesarias en un GPX legítimo).
    """
    if not data or not data.strip():
        raise InvalidGPXError("El archivo está vacío.")
    if b"\x00" in data[:1024] and not data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise InvalidGPXError("El archivo es binario, no un XML/GPX.")

    text = _decode(data)
    if _FORBIDDEN_XML.search(text):
        raise InvalidGPXError("El XML contiene declaraciones DOCTYPE/ENTITY, no permitidas por seguridad.")
    if "<gpx" not in text:
        raise InvalidGPXError("El archivo no contiene un elemento raíz <gpx>.")

    try:
        return gpxpy.parse(text)
    except gpxpy.gpx.GPXXMLSyntaxException as exc:
        raise InvalidGPXError("El archivo no es un XML bien formado.", detail=str(exc)) from exc
    except gpxpy.gpx.GPXException as exc:
        raise InvalidGPXError("El archivo no cumple el formato GPX.", detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - gpxpy puede propagar ValueError/TypeError en datos corruptos
        raise InvalidGPXError("No se pudo interpretar el archivo GPX.", detail=repr(exc)) from exc


def _iter_points(gpx: gpxpy.gpx.GPX) -> list[gpxpy.gpx.GPXTrackPoint | gpxpy.gpx.GPXRoutePoint]:
    """Devuelve los puntos de todos los tracks; si no hay, los de las rutas."""
    points: list[gpxpy.gpx.GPXTrackPoint | gpxpy.gpx.GPXRoutePoint] = [
        p for track in gpx.tracks for segment in track.segments for p in segment.points
    ]
    if not points:
        points = [p for route in gpx.routes for p in route.points]
    return points


def extract_points(gpx: gpxpy.gpx.GPX, settings: Settings | None = None) -> RawTrack:
    """Extrae ``lat``, ``lon``, ``ele`` y ``time`` de todos los puntos del track.

    Limpieza aplicada:
      * se descartan puntos con coordenadas ausentes, no finitas o fuera de rango;
      * se eliminan duplicados consecutivos (paradas con el GPS quieto);
      * las altitudes ausentes se interpolan (o se ponen a 0 si no hay ninguna);
      * los tiempos se descartan en bloque si faltan en algún punto o retroceden.

    Raises:
        InvalidGPXError: menos de dos puntos válidos o demasiados puntos.
    """
    settings = settings or get_settings()
    lat_l: list[float] = []
    lon_l: list[float] = []
    ele_l: list[float] = []
    time_l: list[datetime | None] = []

    for p in _iter_points(gpx):
        lat, lon = p.latitude, p.longitude
        if lat is None or lon is None:
            continue
        try:
            lat_f, lon_f = float(lat), float(lon)
            ele_f = float(p.elevation) if p.elevation is not None else math.nan
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lat_f) and math.isfinite(lon_f)) or abs(lat_f) > 90 or abs(lon_f) > 180:
            continue
        if lat_l and lat_f == lat_l[-1] and lon_f == lon_l[-1]:
            continue
        lat_l.append(lat_f)
        lon_l.append(lon_f)
        ele_l.append(ele_f if math.isfinite(ele_f) and -500.0 <= ele_f <= 9000.0 else math.nan)
        time_l.append(p.time)

    n = len(lat_l)
    if n < 2:
        raise InvalidGPXError("El GPX no contiene un track con al menos dos puntos válidos distintos.")
    if n > settings.max_track_points:
        raise InvalidGPXError(f"El track tiene {n} puntos; el máximo admitido es {settings.max_track_points}.")

    lat = np.asarray(lat_l, dtype=np.float64)
    lon = np.rad2deg(np.unwrap(np.deg2rad(np.asarray(lon_l, dtype=np.float64))))

    # --- Altitud ---------------------------------------------------------
    ele = np.asarray(ele_l, dtype=np.float64)
    valid = np.isfinite(ele)
    has_elevation = bool(valid.any())
    if not has_elevation:
        ele = np.zeros(n)
    elif not valid.all():
        idx = np.arange(n)
        ele = np.interp(idx, idx[valid], ele[valid])

    # --- Tiempo ----------------------------------------------------------
    time_s: FloatArray | None = None
    start_time: datetime | None = None
    if all(t is not None for t in time_l):
        try:
            t0 = time_l[0]
            assert t0 is not None
            candidate = np.asarray([(t - t0).total_seconds() for t in time_l], dtype=np.float64)  # type: ignore[operator]
            if np.all(np.diff(candidate) >= 0) and candidate[-1] > 0:
                time_s, start_time = candidate, t0
            else:
                logger.warning("Marcas de tiempo no monótonas o nulas: se ignoran.")
        except (TypeError, ValueError, OverflowError):  # p. ej. mezcla de fechas con y sin zona horaria
            logger.warning("Marcas de tiempo inconsistentes: se ignoran.")

    return RawTrack(lat=lat, lon=lon, ele=ele, time_s=time_s, has_elevation=has_elevation, start_time=start_time)


# --------------------------------------------------------------------------- #
# Geometría
# --------------------------------------------------------------------------- #
def to_local_xy(lat: FloatArray, lon: FloatArray, lat0: float, lon0: float) -> tuple[FloatArray, FloatArray]:
    """Proyección equirectangular local: grados → metros (Este, Norte) respecto a un origen."""
    x = np.deg2rad(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = np.deg2rad(lat - lat0) * EARTH_RADIUS_M
    return x, y


def to_lat_lon(x: FloatArray, y: FloatArray, lat0: float, lon0: float) -> tuple[FloatArray, FloatArray]:
    """Inversa de :func:`to_local_xy`. La longitud se normaliza a ``[-180, 180)``."""
    lat = lat0 + np.rad2deg(y / EARTH_RADIUS_M)
    lon = lon0 + np.rad2deg(x / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    return lat, (lon + 180.0) % 360.0 - 180.0


def haversine_steps(lat: FloatArray, lon: FloatArray) -> FloatArray:
    """Distancia de círculo máximo (m) entre puntos consecutivos."""
    phi = np.deg2rad(lat)
    dphi = np.diff(phi)
    dlmb = np.deg2rad(np.diff(lon))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi[:-1]) * np.cos(phi[1:]) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# --------------------------------------------------------------------------- #
# Filtrado
# --------------------------------------------------------------------------- #
def _odd_window(requested: int, n: int) -> int:
    """Mayor ventana impar ``<= min(requested, n)``."""
    w = min(requested, n)
    return w if w % 2 == 1 else w - 1


def remove_gps_spikes(
    x: FloatArray, y: FloatArray, *, window: int = 7, min_deviation_m: float = 25.0, n_sigmas: float = 4.0
) -> tuple[FloatArray, FloatArray, int]:
    """Corrige saltos erráticos con un filtro de Hampel bidimensional.

    Un punto se considera salto si su distancia a la mediana móvil supera
    ``max(n_sigmas · σ_robusta, min_deviation_m, 4 · paso mediano)``; en ese
    caso se sustituye por la mediana local. A diferencia de una media móvil,
    no arrastra el error del salto a los puntos vecinos.

    Returns:
        ``(x, y, nº de puntos corregidos)``.
    """
    w = _odd_window(window, x.size)
    if w < 3:
        return x, y, 0
    mx = median_filter(x, size=w, mode="nearest")
    my = median_filter(y, size=w, mode="nearest")
    deviation = np.hypot(x - mx, y - my)
    sigma = 1.4826 * float(np.median(deviation))
    median_step = float(np.median(np.hypot(np.diff(x), np.diff(y))))
    threshold = max(n_sigmas * sigma, min_deviation_m, 4.0 * median_step)
    mask = deviation > threshold
    return np.where(mask, mx, x), np.where(mask, my, y), int(mask.sum())


def smooth_series(values: FloatArray, window: int, polyorder: int = 2) -> FloatArray:
    """Suavizado Savitzky-Golay (preserva curvas y cimas mejor que la media móvil).

    Si la serie es demasiado corta para la ventana, se devuelve sin cambios.
    """
    w = _odd_window(window, values.size)
    if w <= polyorder:
        return values
    return savgol_filter(values, window_length=w, polyorder=polyorder, mode="interp")


def elevation_gain_loss(ele: FloatArray, threshold_m: float) -> tuple[float, float]:
    """Desnivel positivo y negativo acumulado con histéresis.

    Solo se contabiliza un cambio cuando supera ``threshold_m`` respecto al
    último punto de referencia, lo que evita inflar el desnivel con ruido
    barométrico/GPS.
    """
    gain = loss = 0.0
    values: list[float] = ele.tolist()
    ref = values[0]
    for e in values[1:]:
        delta = e - ref
        if delta >= threshold_m:
            gain += delta
            ref = e
        elif delta <= -threshold_m:
            loss -= delta
            ref = e
    return gain, loss


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def estimate_duration_s(distance_m: float, gain_m: float) -> float:
    """Duración estimada sin marcas de tiempo (regla de Naismith: 5 km/h + 1 h por cada 600 m de subida)."""
    return distance_m / (5000.0 / 3600.0) + gain_m / 600.0 * 3600.0


def compute_metrics(
    raw: RawTrack, steps_m: FloatArray, ele_smooth: FloatArray, outliers: int, settings: Settings
) -> TrackMetrics:
    """Calcula las métricas globales del track ya limpio."""
    distance = float(steps_m.sum())
    if raw.has_elevation:
        gain, loss = elevation_gain_loss(ele_smooth, settings.elevation_gain_threshold_m)
    else:
        gain = loss = 0.0

    if raw.time_s is not None:
        duration = float(raw.time_s[-1])
        dt = np.diff(raw.time_s)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dt > 0, steps_m / dt, 0.0)
        moving = float(dt[speed >= settings.moving_speed_threshold_ms].sum()) or duration
        estimated = False
    else:
        duration = moving = estimate_duration_s(distance, gain)
        estimated = True

    def kmh(seconds: float) -> float:
        return distance / seconds * 3.6 if seconds > 0 else 0.0

    return TrackMetrics(
        point_count=len(raw),
        total_distance_m=round(distance, 1),
        elevation_gain_m=round(gain, 1),
        elevation_loss_m=round(loss, 1),
        min_elevation_m=round(float(ele_smooth.min()), 1),
        max_elevation_m=round(float(ele_smooth.max()), 1),
        duration_s=round(duration, 1),
        moving_time_s=round(moving, 1),
        duration_is_estimated=estimated,
        avg_speed_kmh=round(kmh(duration), 2),
        avg_moving_speed_kmh=round(kmh(moving), 2),
        has_elevation=raw.has_elevation,
        has_timestamps=raw.time_s is not None,
        outliers_corrected=outliers,
        start_time=raw.start_time,
    )


# --------------------------------------------------------------------------- #
# Remuestreo para el motor de render
# --------------------------------------------------------------------------- #
def compute_video_duration(distance_m: float, animation_speed: float, settings: Settings) -> float:
    """Duración del vídeo (s): proporcional a la distancia, acotada, y dividida por la velocidad."""
    base = distance_m / 1000.0 * settings.seconds_per_km
    base = min(max(base, settings.min_video_seconds), settings.max_video_seconds)
    return base / animation_speed


def _smoothed_heading(x: FloatArray, y: FloatArray, fps: int) -> FloatArray:
    """Rumbo (0 = norte, horario) filtrado para que la cámara no dé bandazos."""
    dx, dy = np.gradient(x), np.gradient(y)
    sigma = max(fps * 0.75, 1.0)
    # Se suaviza el vector dirección (no el ángulo) para evitar la discontinuidad 359°→0°.
    dx_s = gaussian_filter1d(dx, sigma=sigma, mode="nearest")
    dy_s = gaussian_filter1d(dy, sigma=sigma, mode="nearest")
    return np.rad2deg(np.arctan2(dx_s, dy_s)) % 360.0


def process_gpx(data: bytes, options: RenderOptions, settings: Settings | None = None) -> ProcessedTrack:
    """Pipeline completo: bytes GPX → trayectoria lista para renderizar.

    Args:
        data: contenido en bruto del archivo ``.gpx``.
        options: parámetros de animación (la velocidad determina el nº de fotogramas).
        settings: configuración; por defecto la global.

    Raises:
        InvalidGPXError: GPX corrupto, mal formado o sin recorrido utilizable.
    """
    settings = settings or get_settings()
    raw = extract_points(parse_gpx(data), settings)

    lat0, lon0 = float(raw.lat.mean()), float(raw.lon.mean())
    x, y = to_local_xy(raw.lat, raw.lon, lat0, lon0)

    # 1) Saltos GPS → 2) métricas sobre el track corregido → 3) suavizado fino
    x, y, outliers = remove_gps_spikes(
        x, y, window=settings.outlier_window, min_deviation_m=settings.outlier_min_deviation_m
    )
    clean_lat, clean_lon = to_lat_lon(x, y, lat0, lon0)
    steps = haversine_steps(clean_lat, clean_lon)

    ele = median_filter(raw.ele, size=_odd_window(5, len(raw)), mode="nearest") if len(raw) >= 3 else raw.ele
    ele = smooth_series(ele, settings.elevation_smooth_window)
    metrics = compute_metrics(raw, steps, ele, outliers, settings)
    if metrics.total_distance_m < 1.0:
        raise InvalidGPXError("El track no tiene desplazamiento apreciable (distancia < 1 m).")

    xs = smooth_series(x, settings.position_smooth_window)
    ys = smooth_series(y, settings.position_smooth_window)

    # Remuestreo uniforme por longitud de arco: la cámara avanza a velocidad
    # constante, sin acelerones ni paradas aunque el muestreo GPS sea irregular.
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))))
    keep = np.concatenate(([True], np.diff(arc) > 0))  # np.interp exige abscisas crecientes
    arc_k = arc[keep]
    if arc_k.size < 2:
        raise InvalidGPXError("El track no tiene desplazamiento apreciable tras el suavizado.")

    duration = compute_video_duration(metrics.total_distance_m, options.animation_speed, settings)
    n_frames = max(int(round(duration * settings.fps)), 2)
    target = np.linspace(0.0, arc_k[-1], n_frames)

    fx = np.interp(target, arc_k, xs[keep])
    fy = np.interp(target, arc_k, ys[keep])
    fele = np.interp(target, arc_k, ele[keep])
    real_dist = np.concatenate(([0.0], np.cumsum(steps)))[keep]
    fdist = np.interp(target, arc_k, real_dist)
    if raw.time_s is not None:
        felapsed = np.interp(target, arc_k, raw.time_s[keep])
    else:
        felapsed = target / arc_k[-1] * metrics.duration_s
    flat, flon = to_lat_lon(fx, fy, lat0, lon0)

    return ProcessedTrack(
        metrics=metrics,
        fps=settings.fps,
        video_duration_s=n_frames / settings.fps,
        origin_lat=lat0,
        origin_lon=(lon0 + 180.0) % 360.0 - 180.0,
        lat=flat,
        lon=flon,
        ele=fele,
        x=fx,
        y=fy,
        distance_m=fdist,
        video_t=np.arange(n_frames, dtype=np.float64) / settings.fps,
        elapsed_s=felapsed,
        heading_deg=_smoothed_heading(fx, fy, settings.fps),
    )

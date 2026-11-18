"""Descarga y preparación del terreno: modelo digital de elevaciones + ortofoto satélite.

Fuentes por defecto (sin clave de API, configurables en :mod:`app.config`):
  * Relieve: *Terrain Tiles* de Mapzen/AWS en codificación *terrarium*.
  * Imagen: *Esri World Imagery* (requiere mostrar la atribución en el vídeo).

Las teselas se guardan en una caché en disco, de modo que repetir un render de
la misma zona no vuelve a descargar nada.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import map_coordinates

from .config import Settings
from .deadline import check_deadline
from .exceptions import RenderEngineError
from .gpx_processor import to_lat_lon, to_local_xy

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
TILE = 256
_USER_AGENT = "gpx-render-service/1.1"


# --------------------------------------------------------------------------- #
# Web Mercator
# --------------------------------------------------------------------------- #
def _merc(lat: FloatArray | float, lon: FloatArray | float) -> tuple[FloatArray, FloatArray]:
    """(lat, lon) → coordenadas Web Mercator normalizadas en [0, 1] (y crece hacia el sur)."""
    lat_r = np.deg2rad(np.clip(lat, -85.0, 85.0))
    return (np.asarray(lon, dtype=np.float64) + 180.0) / 360.0, (1.0 - np.arcsinh(np.tan(lat_r)) / math.pi) / 2.0


def _merc_inv(mx: FloatArray, my: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Inversa de :func:`_merc` → (lat, lon)."""
    return np.rad2deg(np.arctan(np.sinh(math.pi * (1.0 - 2.0 * my)))), mx * 360.0 - 180.0


def _pick_zoom(span: float, max_px: int, max_zoom: int) -> int:
    """Mayor zoom cuyo mosaico no supera ``max_px`` píxeles en el lado mayor."""
    for z in range(max_zoom, 0, -1):
        if span * TILE * 2**z <= max_px:
            return z
    return 1


# --------------------------------------------------------------------------- #
# Descarga de teselas
# --------------------------------------------------------------------------- #
def _enforce_tile_cache_limit(cache_dir: Path, max_bytes: int = 2 * 1024**3) -> None:
    """B-6: LRU simple - borra las teselas más antiguas si la caché supera el límite (2GB por defecto)."""
    if not cache_dir.is_dir() or max_bytes <= 0:
        return
    try:
        files = [(p, p.stat()) for p in cache_dir.glob("*.tile")]
        total = sum(s.st_size for _, s in files)
        if total <= max_bytes:
            return
        files.sort(key=lambda x: x[1].st_mtime)  # más antiguas primero
        for p, s in files:
            if total <= max_bytes * 0.8:
                break
            try:
                p.unlink(missing_ok=True)
                total -= s.st_size
            except OSError:
                pass
    except OSError:
        pass


def fetch_tile_bytes(url: str, cache_dir: Path, timeout: float, retries: int = 3) -> bytes:
    """Descarga (o lee de la caché en disco) una tesela y devuelve sus bytes, ya validados como imagen."""
    cached = cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".tile")  # noqa: S324 - no criptográfico
    if cached.is_file():
        try:
            data = cached.read_bytes()
            Image.open(io.BytesIO(data)).verify()
            # tocar mtime para LRU
            cached.touch(exist_ok=True)
            return data
        except Exception:  # noqa: BLE001 - caché corrupta: se vuelve a descargar
            cached.unlink(missing_ok=True)

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                data = resp.read()
            Image.open(io.BytesIO(data)).verify()
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(f".{time.monotonic_ns()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(cached)  # escritura atómica: seguro con varios renders a la vez
            # B-6: verificar límite de caché de forma oportunista (cada 100 descargas)
            if hash(url) % 100 == 0:
                _enforce_tile_cache_limit(cache_dir)
            return data
        except Exception as exc:  # noqa: BLE001 - red, tiempo de espera o respuesta que no es una imagen
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise RenderEngineError("No se pudo descargar una tesela del mapa.", detail=f"{url}: {last!r}")


def _fetch_tile(url: str, cache_dir: Path, timeout: float) -> Image.Image:
    """Tesela como imagen RGB."""
    return Image.open(io.BytesIO(fetch_tile_bytes(url, cache_dir, timeout))).convert("RGB")


def _fetch_mosaic(
    template: str, zoom: int, bounds: tuple[float, float, float, float], settings: Settings
) -> tuple[np.ndarray, float, float]:
    """Descarga las teselas que cubren ``bounds`` y las une en un mosaico.

    Args:
        bounds: ``(mx_oeste, my_norte, mx_este, my_sur)`` en Mercator normalizado.

    Returns:
        ``(mosaico RGB uint8, píxel global X del borde izquierdo, píxel global Y del borde superior)``.
    """
    n = 2**zoom
    x0, y0 = int(bounds[0] * n), int(bounds[1] * n)
    x1, y1 = int(bounds[2] * n), int(bounds[3] * n)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, n - 1), min(y1, n - 1)
    jobs = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    if len(jobs) > settings.max_tiles_per_layer:
        raise RenderEngineError(f"La zona del recorrido requiere demasiadas teselas ({len(jobs)}).")

    settings.tile_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Descargando %d teselas (zoom %d) de %s", len(jobs), zoom, template.split("/")[2])
    with ThreadPoolExecutor(max_workers=settings.tile_download_workers) as pool:
        tiles = list(
            pool.map(
                lambda t: _fetch_tile(
                    template.format(z=zoom, x=t[0], y=t[1]), settings.tile_cache_dir, settings.tile_timeout_s
                ),
                jobs,
            )
        )
    mosaic = np.zeros(((y1 - y0 + 1) * TILE, (x1 - x0 + 1) * TILE, 3), dtype=np.uint8)
    for (tx, ty), tile in zip(jobs, tiles):
        if tile.size != (TILE, TILE):
            tile = tile.resize((TILE, TILE), Image.Resampling.BILINEAR)
        mosaic[(ty - y0) * TILE : (ty - y0 + 1) * TILE, (tx - x0) * TILE : (tx - x0 + 1) * TILE] = np.asarray(tile)
    return mosaic, float(x0 * TILE), float(y0 * TILE)


# --------------------------------------------------------------------------- #
# Terreno
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TerrainData:
    """Malla regular del terreno en coordenadas locales (metros) con su textura."""

    x: FloatArray  # (filas, columnas) Este local; la fila 0 es el borde norte
    y: FloatArray  # (filas, columnas) Norte local
    z: FloatArray  # (filas, columnas) altitud en metros
    u: FloatArray  # (columnas,) coordenada de textura horizontal
    v: FloatArray  # (filas,) coordenada de textura vertical (0 = fila superior de la imagen)
    texture: np.ndarray  # (alto, ancho, 3) uint8
    attribution: str
    _bounds: tuple[float, float, float, float]
    _origin: tuple[float, float]

    def height_at(self, x: FloatArray, y: FloatArray) -> FloatArray:
        """Altitud del terreno (interpolación bilineal de la malla) en puntos locales (x, y)."""
        lat, lon = to_lat_lon(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), *self._origin)
        mx, my = _merc(lat, lon)
        rows, cols = self.z.shape
        col = (mx - self._bounds[0]) / (self._bounds[2] - self._bounds[0]) * (cols - 1)
        row = (my - self._bounds[1]) / (self._bounds[3] - self._bounds[1]) * (rows - 1)
        return map_coordinates(self.z, [row, col], order=1, mode="nearest")


def build_terrain(
    track_x: FloatArray, track_y: FloatArray, origin_lat: float, origin_lon: float, margin_m: float, settings: Settings
) -> TerrainData:
    """Construye el terreno que rodea al recorrido.

    Args:
        track_x, track_y: recorrido en metros locales (misma proyección que el payload).
        origin_lat, origin_lon: origen de la proyección local.
        margin_m: margen alrededor del recorrido.

    Raises:
        RenderEngineError: sin conexión, servidor de teselas caído o zona demasiado extensa.
    """
    corners_x = np.array([track_x.min() - margin_m, track_x.max() + margin_m])
    corners_y = np.array([track_y.max() + margin_m, track_y.min() - margin_m])  # norte, sur
    lat, lon = to_lat_lon(corners_x, corners_y, origin_lat, origin_lon)
    if lon[0] > lon[1]:
        raise RenderEngineError("Los recorridos que cruzan el antimeridiano no están soportados en 3D.")
    mx, my = _merc(lat, lon)
    bounds = (float(mx[0]), float(my[0]), float(mx[1]), float(my[1]))
    span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

    # --- Malla y elevaciones -----------------------------------------------
    aspect = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0])
    cols = settings.terrain_mesh_max if aspect <= 1 else max(int(settings.terrain_mesh_max / aspect), 2)
    rows = settings.terrain_mesh_max if aspect > 1 else max(int(settings.terrain_mesh_max * aspect), 2)
    gmx, gmy = np.meshgrid(np.linspace(bounds[0], bounds[2], cols), np.linspace(bounds[1], bounds[3], rows))

    dem_zoom = _pick_zoom(span, settings.dem_max_px, settings.dem_max_zoom)
    check_deadline()  # la descarga de teselas es la fase más lenta y depende de la red
    dem_rgb, dem_px0, dem_py0 = _fetch_mosaic(settings.dem_url, dem_zoom, bounds, settings)
    dem = dem_rgb[..., 0].astype(np.float64) * 256.0 + dem_rgb[..., 1] + dem_rgb[..., 2] / 256.0 - 32768.0
    scale = TILE * 2**dem_zoom
    z = map_coordinates(dem, [gmy * scale - dem_py0 - 0.5, gmx * scale - dem_px0 - 0.5], order=1, mode="nearest")
    z = np.maximum(z, 0.0)  # la batimetría convertiría el mar en una fosa

    glat, glon = _merc_inv(gmx, gmy)
    gx, gy = to_local_xy(glat, glon, origin_lat, origin_lon)

    # --- Ortofoto ----------------------------------------------------------
    img_zoom = _pick_zoom(span, settings.imagery_max_px, settings.imagery_max_zoom)
    check_deadline()
    img, img_px0, img_py0 = _fetch_mosaic(settings.imagery_url, img_zoom, bounds, settings)
    scale = TILE * 2**img_zoom
    u = (np.linspace(bounds[0], bounds[2], cols) * scale - img_px0) / img.shape[1]
    v = (np.linspace(bounds[1], bounds[3], rows) * scale - img_py0) / img.shape[0]

    logger.info(
        "Terreno listo: malla %dx%d, DEM z%d, imagen z%d (%dx%d px), altitudes %.0f–%.0f m",
        cols, rows, dem_zoom, img_zoom, img.shape[1], img.shape[0], z.min(), z.max(),
    )  # fmt: skip
    return TerrainData(
        x=gx, y=gy, z=z, u=u, v=v, texture=img, attribution=settings.imagery_attribution,
        _bounds=bounds, _origin=(origin_lat, origin_lon),
    )  # fmt: skip

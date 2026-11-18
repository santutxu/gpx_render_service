"""Momentos: fotos y clips que el usuario ancla a un punto de la ruta.

Al llegar el recorrido a ese punto, la animación 3D se detiene, el fotograma
actual se desenfoca y sobre él aparece la foto (unos segundos) o el clip,
con su pie de foto; después el vuelo continúa.

Este módulo se encarga de:
  1. validar y sanear los archivos subidos (:func:`prepare_moments`);
  2. situar cada momento en el fotograma de la ruta que le corresponde;
  3. generar los fotogramas intercalados (:class:`MomentRenderer`).

Seguridad: el tipo de archivo se decide por sus *bytes mágicos*, nunca por la
extensión ni por el ``Content-Type``. Las imágenes se recodifican con Pillow
y los vídeos se abren forzando el demultiplexor (``-f mov`` / ``-f matroska``)
para que FFmpeg no pueda interpretar el archivo como una lista de reproducción
que lea otros ficheros o URL.
"""

from __future__ import annotations

import io
import json
import logging
import re
import shutil
import subprocess
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from pydantic import ValidationError

from .config import Settings
from .exceptions import GPXRenderError
from .gpx_processor import EARTH_RADIUS_M, ProcessedTrack
from .schemas import MomentSpec

logger = logging.getLogger(__name__)

_MP4_FORMAT = "mov,mp4,m4a,3gp,3g2,mj2"
_MKV_FORMAT = "matroska,webm"
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif", b"avis"}
_QUICKTIME_ATOMS = {b"ftyp", b"moov", b"wide", b"mdat", b"free", b"skip"}
_DURATION_REGEX = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_MAX_IMAGE_SIDE = 2560
#: Un momento debe estar, como mucho, a esta distancia de la ruta.
_MAX_OFF_ROUTE_M = 500.0


class InvalidMomentError(GPXRenderError):
    """Los datos o el archivo de un momento no son válidos."""

    status_code = 422
    error_code = "invalid_moment"


@dataclass(frozen=True)
class PreparedMoment:
    """Momento validado y listo para insertarse en el vídeo."""

    frame: int  # fotograma de la ruta tras el que se muestra
    kind: str  # "image" | "video"
    path: Path
    caption: str
    distance_m: float
    width: int  # tamaño del medio tal como se ve (con la rotación aplicada)
    height: int
    duration_s: float
    input_format: str | None = None  # demultiplexor de FFmpeg (solo vídeo)


# --------------------------------------------------------------------------- #
# Validación de la petición
# --------------------------------------------------------------------------- #
def parse_moment_specs(raw: str | None, file_count: int, settings: Settings) -> list[MomentSpec]:
    """Valida el campo ``moments`` (JSON) del formulario.

    Raises:
        InvalidMomentError: JSON mal formado, demasiados momentos o referencias a archivos inexistentes.
    """
    if raw is None or not raw.strip():
        return []
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            raise ValueError("se esperaba una lista")
        specs = [MomentSpec.model_validate(item) for item in items]
    except (ValueError, ValidationError) as exc:
        raise InvalidMomentError("El campo 'moments' no es válido.", detail=str(exc)) from exc
    if len(specs) > settings.max_moments:
        raise InvalidMomentError(f"Se admiten como máximo {settings.max_moments} momentos por vídeo.")
    used = [s.file_index for s in specs]
    if any(i >= file_count for i in used) or len(set(used)) != len(used):
        raise InvalidMomentError("Cada momento debe llevar su propio archivo adjunto.")
    return specs


def _sniff(head: bytes) -> tuple[str, str | None]:
    """Identifica el tipo real del archivo: ``("image", None)`` o ``("video", formato FFmpeg)``."""
    if head[:3] == b"\xff\xd8\xff" or head[:8] == b"\x89PNG\r\n\x1a\n" or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return "image", None
    if head[4:8] == b"ftyp" and head[8:12] in _HEIF_BRANDS:
        raise InvalidMomentError("Las fotos HEIC/AVIF no están admitidas: conviértela a JPG o PNG.")
    if head[4:8] in _QUICKTIME_ATOMS:
        return "video", _MP4_FORMAT
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "video", _MKV_FORMAT
    raise InvalidMomentError("Archivo no admitido. Usa una foto JPG, PNG o WebP, o un clip MP4, MOV o WebM.")


def _prepare_image(raw: Path, dest: Path) -> tuple[int, int]:
    """Recodifica la foto a JPEG (sin metadatos, orientación aplicada, tamaño acotado). B-7: limita bomba de descompresión."""
    # B-7: Pillow por defecto permite 178M píxeles (~500MB RAM); lo bajamos y usamos draft
    Image.MAX_IMAGE_PIXELS = 50_000_000  # ~50 MP, suficiente para fotos 8K
    try:
        with Image.open(raw) as img:
            # draft reduce memoria antes de decodificar (para JPEG)
            if hasattr(img, "draft"):
                try:
                    img.draft("RGB", (_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
                except Exception:
                    pass
            # verifica que la imagen no sea una bomba antes de decodificar completa
            w, h = img.size
            if w * h > Image.MAX_IMAGE_PIXELS:
                raise InvalidMomentError(f"La imagen es demasiado grande ({w}×{h} px). Máximo {Image.MAX_IMAGE_PIXELS} píxeles.")
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            img.save(dest, format="JPEG", quality=92)
            return img.size
    except InvalidMomentError:
        raise
    except Exception as exc:  # noqa: BLE001 - imagen corrupta, truncada
        raise InvalidMomentError("La imagen está dañada o no se puede leer.", detail=repr(exc)) from exc


def _prepare_video(raw: Path, dest_dir: Path, index: int, fmt: str, settings: Settings) -> tuple[int, int, float]:
    """Comprueba que el clip se puede decodificar y obtiene su tamaño en pantalla y su duración."""
    ffmpeg = shutil.which(settings.ffmpeg_binary)
    if ffmpeg is None:
        raise GPXRenderError("FFmpeg no está disponible para procesar el clip.")
    first = dest_dir / f"first_{index}.png"
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-f", fmt, "-i", str(raw),
           "-map", "0:v:0", "-frames:v", "1", "-c:v", "png", "-f", "image2", str(first)]  # fmt: skip
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=60, check=False)  # noqa: S603
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise InvalidMomentError("No se pudo procesar el clip de vídeo.", detail=repr(exc)) from exc
    if result.returncode != 0 or not first.is_file():
        raise InvalidMomentError("El clip de vídeo está dañado o usa un formato no admitido.", detail=result.stderr[-800:])
    try:
        with Image.open(first) as img:
            width, height = img.size  # FFmpeg ya aplica la rotación de los vídeos de móvil
    except Exception as exc:  # noqa: BLE001
        raise InvalidMomentError("No se pudo leer el clip de vídeo.", detail=repr(exc)) from exc
    finally:
        first.unlink(missing_ok=True)
    duration = settings.moment_clip_max_seconds
    if match := _DURATION_REGEX.search(result.stderr):
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return width, height, max(min(duration, settings.moment_clip_max_seconds), 0.5)


def _assign_frame(spec: MomentSpec, track: ProcessedTrack) -> int:
    """Fotograma de la ruta más cercano al punto marcado.

    Si la ruta pasa dos veces por el mismo sitio (ida y vuelta), ``position``
    (fracción de la distancia total) indica cuál de las pasadas eligió el usuario.
    """
    lat0 = np.deg2rad(spec.lat)
    dx = np.deg2rad(track.lon - spec.lon) * np.cos(lat0)
    dx = (dx + np.pi) % (2 * np.pi) - np.pi  # antimeridiano
    dist = EARTH_RADIUS_M * np.hypot(dx, np.deg2rad(track.lat) - lat0)
    if spec.position is not None:
        fraction = track.distance_m / max(float(track.distance_m[-1]), 1.0)
        window = np.abs(fraction - spec.position) <= 0.12
        if window.any() and float(dist[window].min()) <= _MAX_OFF_ROUTE_M:
            dist = np.where(window, dist, np.inf)
    frame = int(np.argmin(dist))
    if not np.isfinite(dist[frame]) or dist[frame] > _MAX_OFF_ROUTE_M:
        raise InvalidMomentError("Un momento está demasiado lejos de la ruta. Márcalo sobre el recorrido.")
    return frame


def prepare_moments(
    specs: list[MomentSpec], raw_files: dict[int, Path], media_dir: Path, track: ProcessedTrack, settings: Settings
) -> list[PreparedMoment]:
    """Valida los archivos y sitúa cada momento en la ruta.

    Args:
        specs: momentos pedidos por el usuario.
        raw_files: archivo subido (sin procesar) de cada ``file_index``.
        media_dir: directorio temporal del trabajo donde dejar los medios saneados.

    Returns:
        Momentos ordenados por su posición en la ruta.

    Raises:
        InvalidMomentError: archivo no admitido o dañado, o punto fuera de la ruta.
    """
    prepared: list[PreparedMoment] = []
    for number, spec in enumerate(specs):
        raw = raw_files[spec.file_index]
        with raw.open("rb") as fh:
            kind, fmt = _sniff(fh.read(16))
        frame = _assign_frame(spec, track)
        caption = " ".join(spec.caption.split())
        if kind == "image":
            path = media_dir / f"moment_{number}.jpg"
            width, height = _prepare_image(raw, path)
            raw.unlink(missing_ok=True)
            duration = settings.moment_image_seconds
        else:
            assert fmt is not None
            width, height, duration = _prepare_video(raw, media_dir, number, fmt, settings)
            path = raw
        prepared.append(
            PreparedMoment(frame, kind, path, caption, float(track.distance_m[frame]), width, height, duration, fmt)
        )
    prepared.sort(key=lambda m: m.frame)
    return prepared


# --------------------------------------------------------------------------- #
# Generación de fotogramas
# --------------------------------------------------------------------------- #
def _encode(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


class MomentRenderer:
    """Genera los fotogramas PNG de un momento sobre el fotograma 3D en el que se detiene la ruta."""

    FADE_IN = 10
    FADE_OUT = 8

    def __init__(self, width: int, height: int, fps: int, settings: Settings) -> None:
        self._w, self._h, self._fps = width, height, fps
        self._settings = settings
        self._caption_font = ImageFont.load_default(size=max(round(height * 0.034), 12))
        self._label_font = ImageFont.load_default(size=max(round(height * 0.024), 10))

    def _layout(self, background: Image.Image, moment: PreparedMoment) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """Fondo desenfocado + tarjeta blanca tipo instantánea. Devuelve la base y el hueco del medio."""
        w, h = self._w, self._h
        small = background.resize((max(w // 4, 1), max(h // 4, 1)), Image.Resampling.BILINEAR)
        base = small.filter(ImageFilter.GaussianBlur(5)).resize((w, h), Image.Resampling.BILINEAR)
        base = ImageEnhance.Brightness(base).enhance(0.45)

        border = max(round(h * 0.012), 4)
        caption_h = round(h * 0.075) if moment.caption else 0
        scale = min(w * 0.78 / moment.width, (h * 0.70 - caption_h) / moment.height)
        mw = max(int(moment.width * scale) // 2 * 2, 2)  # par: lo exige el escalado YUV de FFmpeg
        mh = max(int(moment.height * scale) // 2 * 2, 2)
        cw, ch = mw + 2 * border, mh + 2 * border + caption_h
        cx, cy = (w - cw) // 2, (h - ch) // 2

        draw = ImageDraw.Draw(base, "RGBA")
        draw.rounded_rectangle((cx - 2, cy + border, cx + cw + 2, cy + ch + border * 2), border * 2, fill=(0, 0, 0, 110))
        draw.rounded_rectangle((cx, cy, cx + cw, cy + ch), border, fill=(250, 250, 248, 255))
        label = f"KM {moment.distance_m / 1000:.1f}".replace(".", ",")
        draw.text((cx + 2, cy - self._label_font.size * 1.7), label, font=self._label_font, fill=(255, 255, 255, 235))
        if moment.caption:
            text = moment.caption
            while len(text) > 1 and draw.textlength(text, font=self._caption_font) > mw:
                text = text[:-2].rstrip() + "…"
            tw = draw.textlength(text, font=self._caption_font)
            ty = cy + border + mh + (caption_h + border - self._caption_font.size * 1.25) / 2
            draw.text((cx + (cw - tw) / 2, ty), text, font=self._caption_font, fill=(28, 30, 34, 255))
        return base, (cx + border, cy + border, mw, mh)

    def frames(self, moment: PreparedMoment, background_png: bytes) -> Iterator[bytes]:
        """Fotogramas PNG del momento: entrada con fundido, exposición y salida con fundido.

        Nunca lanza por un medio problemático: si el clip no se puede leer en este
        punto, el momento se omite y el vídeo de la ruta continúa.
        """
        try:
            with Image.open(io.BytesIO(background_png)) as img:
                original = img.convert("RGB")
            if original.size != (self._w, self._h):
                original = original.resize((self._w, self._h))
            base, box = self._layout(original, moment)
            source = self._image_frames(moment, base, box) if moment.kind == "image" else self._video_frames(moment, base, box)
            yield from self._with_fades(original, source)
        except Exception:  # noqa: BLE001
            logger.exception("No se pudo insertar el momento del fotograma %d; se omite", moment.frame)

    def _image_frames(self, moment: PreparedMoment, base: Image.Image, box: tuple[int, int, int, int]) -> Iterator[Image.Image]:
        x, y, mw, mh = box
        with Image.open(moment.path) as img:
            base.paste(img.convert("RGB").resize((mw, mh), Image.Resampling.LANCZOS), (x, y))
        total = max(round(moment.duration_s * self._fps), self.FADE_IN + self.FADE_OUT + 1)
        for _ in range(total):
            yield base  # misma imagen: _with_fades la codifica una sola vez

    def _video_frames(self, moment: PreparedMoment, base: Image.Image, box: tuple[int, int, int, int]) -> Iterator[Image.Image]:
        x, y, mw, mh = box
        ffmpeg = shutil.which(self._settings.ffmpeg_binary)
        if ffmpeg is None or moment.input_format is None:
            return
        cmd = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-f", moment.input_format, "-i", str(moment.path),
               "-map", "0:v:0", "-t", f"{moment.duration_s:.3f}", "-vf", f"fps={self._fps},scale={mw}:{mh}",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "-an", "pipe:1"]  # fmt: skip
        size = mw * mh * 3
        limit = round(self._settings.moment_clip_max_seconds * self._fps)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)  # noqa: S603
        try:
            assert proc.stdout is not None
            for _ in range(limit):
                data = proc.stdout.read(size)
                if len(data) < size:
                    break
                frame = base.copy()
                frame.paste(Image.frombytes("RGB", (mw, mh), data), (x, y))
                yield frame
        finally:
            proc.kill()
            proc.wait()

    def _with_fades(self, original: Image.Image, source: Iterator[Image.Image]) -> Iterator[bytes]:
        """Aplica el fundido de entrada a los primeros fotogramas y el de salida a los últimos.

        La longitud de un clip no se conoce de antemano, así que se retienen los
        últimos ``FADE_OUT`` fotogramas hasta saber que el flujo ha terminado.
        """
        tail: deque[Image.Image] = deque()
        cache: tuple[int, bytes] | None = None  # evita recodificar una foto fija en cada fotograma
        index = 0
        for frame in source:
            tail.append(frame)
            if len(tail) <= self.FADE_OUT:
                continue
            current = tail.popleft()
            if index < self.FADE_IN:
                yield _encode(Image.blend(original, current, (index + 1) / (self.FADE_IN + 1)))
            elif cache is not None and cache[0] == id(current):
                yield cache[1]
            else:
                cache = (id(current), _encode(current))
                yield cache[1]
            index += 1
        count = len(tail)
        for j, frame in enumerate(tail):
            yield _encode(Image.blend(original, frame, 1.0 - (j + 1) / (count + 1)))

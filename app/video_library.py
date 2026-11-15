"""Biblioteca de vídeos generados: catálogo persistente en disco.

Cada vídeo terminado ``<id>.mp4`` va acompañado de un ``<id>.json`` con sus
metadatos. Al vivir en disco (y no en la memoria del proceso), los vídeos
siguen siendo listables, reproducibles y borrables tras reiniciar el servidor.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .exceptions import GPXRenderError
from .schemas import RenderOptions, StoredVideo, TrackMetrics

logger = logging.getLogger(__name__)

_ID_REGEX = re.compile(r"^[0-9a-f]{32}$")


class VideoNotFoundError(GPXRenderError):
    """No existe ningún vídeo con ese identificador."""

    status_code = 404
    error_code = "video_not_found"


class VideoLibrary:
    """Operaciones de consulta y borrado sobre el directorio de salida."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._lock = threading.Lock()

    @staticmethod
    def is_valid_id(video_id: str) -> bool:
        """Solo se aceptan UUID en hexadecimal: impide rutas como ``../../etc/passwd``."""
        return bool(_ID_REGEX.match(video_id))

    def _paths(self, video_id: str) -> tuple[Path, Path]:
        if not self.is_valid_id(video_id):
            raise VideoNotFoundError("Vídeo no encontrado.")
        return self._dir / f"{video_id}.mp4", self._dir / f"{video_id}.json"

    def register(
        self,
        video_id: str,
        video_path: Path,
        *,
        owner: str | None,
        title: str,
        options: RenderOptions,
        metrics: TrackMetrics,
        duration_s: float,
        width: int,
        height: int,
    ) -> StoredVideo:
        """Guarda los metadatos de un vídeo recién compilado (escritura atómica)."""
        _, meta_path = self._paths(video_id)
        record = StoredVideo(
            video_id=video_id,
            owner=owner,
            title=title,
            created_at=datetime.now(timezone.utc),
            size_bytes=video_path.stat().st_size,
            duration_s=duration_s,
            width=width,
            height=height,
            options=options,
            metrics=metrics,
        )
        tmp = meta_path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(record.model_dump_json(), encoding="utf-8")
            os.replace(tmp, meta_path)
        return record

    def get(self, video_id: str) -> StoredVideo:
        """Metadatos de un vídeo.

        Raises:
            VideoNotFoundError: id inválido, vídeo inexistente o metadatos ilegibles.
        """
        video_path, meta_path = self._paths(video_id)
        if not video_path.is_file():
            raise VideoNotFoundError("Vídeo no encontrado.")
        try:
            return StoredVideo.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, json.JSONDecodeError) as exc:
            raise VideoNotFoundError("Vídeo no encontrado.", detail=repr(exc)) from exc

    def path(self, video_id: str) -> Path:
        """Ruta del MP4 (verificando que existe)."""
        video_path, _ = self._paths(video_id)
        if not video_path.is_file():
            raise VideoNotFoundError("Vídeo no encontrado.")
        return video_path

    def list(self) -> list[StoredVideo]:
        """Todos los vídeos disponibles, del más reciente al más antiguo."""
        videos: list[StoredVideo] = []
        if not self._dir.is_dir():
            return videos
        for meta_path in self._dir.glob("*.json"):
            try:
                videos.append(self.get(meta_path.stem))
            except VideoNotFoundError:
                logger.warning("Metadatos huérfanos o corruptos ignorados: %s", meta_path.name)
        videos.sort(key=lambda v: v.created_at, reverse=True)
        return videos

    def delete(self, video_id: str) -> None:
        """Borra el MP4 y sus metadatos.

        Raises:
            VideoNotFoundError: si el vídeo no existe.
            GPXRenderError: si el sistema de archivos impide el borrado.
        """
        video_path, meta_path = self._paths(video_id)
        with self._lock:
            if not video_path.is_file():
                raise VideoNotFoundError("Vídeo no encontrado.")
            try:
                video_path.unlink()
                meta_path.unlink(missing_ok=True)
            except OSError as exc:
                raise GPXRenderError("No se pudo borrar el vídeo.", detail=str(exc)) from exc
        logger.info("Vídeo %s borrado", video_id)

"""Configuración común de las pruebas."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Iterator

# Asegurar que ffmpeg/ffprobe de ~/bin (persistente) estén en PATH para los tests
# (el binario se descarga vía johnvansickle y sobrevive a reinicios; imageio-ffmpeg no persiste)
_bin = str(pathlib.Path.home() / "bin")
if _bin not in os.environ.get("PATH", ""):
    # inyectar al inicio para que shutil.which y subprocess lo encuentren
    os.environ["PATH"] = _bin + ":" + os.environ.get("PATH", "")
    # también /home/user/bin explícito por si HOME no es /home/user
    alt = "/home/user/bin"
    if alt != _bin and alt not in os.environ["PATH"]:
        os.environ["PATH"] = alt + ":" + os.environ["PATH"]

import pytest


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    """Devuelve el servicio a un estado conocido antes y después de cada prueba.

    Los renders ya no terminan dentro de la petición, así que hay que esperar a
    la cola; y el almacén de trabajos y la biblioteca de vídeos son singletons del
    proceso: sin limpieza, los vídeos de una prueba contaminan los recuentos de
    las siguientes (p. ej. ``video_count`` en la gestión de usuarios).
    """
    from app.main import library, pipeline, store

    for _ in range(2):
        pipeline.join(timeout=120)
        for job in store.list():
            store.remove(job.job_id)
        for video in library.list():
            library.delete(video.video_id)
    yield
    for _ in range(2):
        pipeline.join(timeout=120)
        for job in store.list():
            store.remove(job.job_id)
        for video in library.list():
            library.delete(video.video_id)

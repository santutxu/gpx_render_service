"""Ensamblaje de la secuencia de fotogramas en MP4 mediante FFmpeg."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .exceptions import FFmpegNotFoundError, VideoCompilationError

logger = logging.getLogger(__name__)

_FRAME_REGEX = re.compile(r"^frame_(\d{4,})\.png$")
FRAME_PATTERN = "frame_%04d.png"


def find_ffmpeg(binary: str = "ffmpeg") -> str:
    """Localiza el ejecutable de FFmpeg.

    Raises:
        FFmpegNotFoundError: si no está instalado o no es ejecutable.
    """
    path = shutil.which(binary)
    if path is None:
        raise FFmpegNotFoundError(f"FFmpeg no está disponible ('{binary}' no se encuentra en el PATH).")
    return path


def _count_frames(frames_dir: Path) -> int:
    """Valida que hay una secuencia contigua desde ``frame_0000.png`` y devuelve su tamaño."""
    if not frames_dir.is_dir():
        raise VideoCompilationError(f"El directorio de fotogramas no existe: {frames_dir}")
    indices = sorted(int(m.group(1)) for p in frames_dir.iterdir() if (m := _FRAME_REGEX.match(p.name)))
    if not indices:
        raise VideoCompilationError("No hay fotogramas que ensamblar.")
    if indices != list(range(len(indices))):
        raise VideoCompilationError("La secuencia de fotogramas tiene huecos o no empieza en frame_0000.png.")
    return len(indices)


def build_ffmpeg_command(
    ffmpeg: str, frames_dir: Path, output: Path, *, fps: int, width: int, height: int, crf: int, preset: str
) -> list[str]:
    """Construye la línea de comandos de FFmpeg (como lista: sin shell, sin inyección)."""
    # libx264 + yuv420p exige dimensiones pares; se ajusta sin deformar y se rellena con negro.
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    )
    return [
        ffmpeg,
        "-hide_banner", "-nostdin", "-loglevel", "error",
        "-y",
        "-framerate", str(fps),
        "-start_number", "0",
        "-i", str(frames_dir / FRAME_PATTERN),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-movflags", "+faststart",  # moov al inicio: reproducción progresiva en web
        "-an",
        "-f", "mp4",
        str(output),
    ]  # fmt: skip


def compile_video(
    frames_dir: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: int = 30,
    crf: int = 20,
    preset: str = "medium",
    ffmpeg_binary: str = "ffmpeg",
    timeout_s: int = 1800,
    cleanup: bool = True,
) -> Path:
    """Convierte ``frame_%04d.png`` en un MP4 H.264 (``libx264``, ``yuv420p``).

    El vídeo se escribe primero en un archivo ``.part`` y se renombra de forma
    atómica al terminar, de modo que ``output_path`` nunca contiene un MP4 a
    medias. Los fotogramas solo se borran si la compilación ha tenido éxito
    (en caso de error se conservan para poder diagnosticar o reintentar).

    Args:
        frames_dir: directorio con la secuencia de fotogramas.
        output_path: ruta final del ``.mp4``.
        width, height: resolución de salida (deben ser pares).
        fps: fotogramas por segundo de entrada y salida.
        crf: calidad H.264 (0–51; menor = mejor calidad y más peso).
        preset: compromiso velocidad/compresión de x264.
        ffmpeg_binary: nombre o ruta del ejecutable.
        timeout_s: tiempo máximo de ejecución.
        cleanup: borrar ``frames_dir`` tras el éxito.

    Returns:
        ``output_path``.

    Raises:
        FFmpegNotFoundError: FFmpeg no instalado.
        VideoCompilationError: secuencia inválida, error de FFmpeg, timeout o salida vacía.
    """
    if width % 2 or height % 2:
        raise VideoCompilationError(f"La resolución debe ser par para yuv420p (recibido {width}x{height}).")
    ffmpeg = find_ffmpeg(ffmpeg_binary)
    n_frames = _count_frames(frames_dir)

    partial = output_path.with_name(output_path.name + ".part")
    cmd = build_ffmpeg_command(
        ffmpeg, frames_dir, partial, fps=fps, width=width, height=height, crf=crf, preset=preset
    )
    logger.info("Compilando %d fotogramas → %s", n_frames, output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(  # noqa: S603 - lista de argumentos, sin shell
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout_s, check=False
        )
        if result.returncode != 0:
            raise VideoCompilationError(
                f"FFmpeg terminó con código {result.returncode}.", detail=result.stderr.strip()[-2000:]
            )
        if not partial.is_file() or partial.stat().st_size == 0:
            raise VideoCompilationError("FFmpeg no generó ningún archivo de salida.")
        os.replace(partial, output_path)
    except subprocess.TimeoutExpired as exc:  # subprocess.run ya ha matado el proceso
        raise VideoCompilationError(f"FFmpeg superó el tiempo límite de {timeout_s} s.") from exc
    except OSError as exc:
        raise VideoCompilationError("Error de sistema al ejecutar FFmpeg.", detail=str(exc)) from exc
    finally:
        partial.unlink(missing_ok=True)

    if cleanup:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return output_path


class StreamingVideoCompiler:
    """Codifica el vídeo de forma incremental mientras el motor sigue renderizando.

    Los fotogramas de un vídeo 1080p con imagen satélite pesan varios MB cada
    uno: guardar la secuencia completa antes de compilar puede ocupar varios GB
    (y ``/tmp`` suele ser un ``tmpfs`` en RAM). Esta clase mantiene un proceso
    FFmpeg abierto (``image2pipe``), le entrega cada ``frame_XXXX.png`` en cuanto
    está terminado y lo borra del disco, de modo que el directorio temporal
    nunca contiene más que unos pocos fotogramas.

    Uso::

        with StreamingVideoCompiler(out, width=1920, height=1080) as compiler:
            ...                                  # el motor escribe frame_0000.png, frame_0001.png…
            compiler.feed(frames_dir, upto=n)    # consume y borra los fotogramas [siguiente, n)
            compiler.finish(expected=total)

    Si el bloque termina con una excepción, FFmpeg se detiene y el vídeo parcial se elimina.
    """

    def __init__(
        self,
        output_path: Path,
        *,
        width: int,
        height: int,
        fps: int = 30,
        crf: int = 20,
        preset: str = "medium",
        ffmpeg_binary: str = "ffmpeg",
        timeout_s: int = 1800,
    ) -> None:
        if width % 2 or height % 2:
            raise VideoCompilationError(f"La resolución debe ser par para yuv420p (recibido {width}x{height}).")
        self._output = output_path
        self._partial = output_path.with_name(output_path.name + ".part")
        self._timeout = timeout_s
        self._next = 0
        self._finished = False
        cmd = build_ffmpeg_command(
            find_ffmpeg(ffmpeg_binary), Path("."), self._partial, fps=fps, width=width, height=height, crf=crf, preset=preset
        )
        # Sustituye la entrada por secuencia de archivos por una tubería de PNG
        i = cmd.index("-start_number")
        self._cmd = cmd[:i] + ["-f", "image2pipe", "-c:v", "png", "-i", "pipe:0"] + cmd[i + 4 :]
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 - se cierra en __exit__

    @property
    def frames_fed(self) -> int:
        """Número de fotogramas entregados a FFmpeg."""
        return self._next

    def __enter__(self) -> StreamingVideoCompiler:
        try:
            self._output.parent.mkdir(parents=True, exist_ok=True)
            self._proc = subprocess.Popen(  # noqa: S603 - lista de argumentos, sin shell
                self._cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr
            )
        except OSError as exc:
            self._stderr.close()
            raise VideoCompilationError("No se pudo iniciar FFmpeg.", detail=str(exc)) from exc
        return self

    def _error_output(self) -> str:
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace").strip()[-2000:]

    def feed(self, frames_dir: Path, upto: int) -> None:
        """Envía a FFmpeg los fotogramas ``[siguiente, upto)`` y los borra del disco.

        Raises:
            VideoCompilationError: falta un fotograma, no es PNG o FFmpeg ha muerto.
        """
        assert self._proc is not None and self._proc.stdin is not None, "usar dentro de un bloque 'with'"
        while self._next < upto:
            path = frames_dir / f"frame_{self._next:04d}.png"
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise VideoCompilationError(f"Falta el fotograma {path.name} en la secuencia.") from exc
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise VideoCompilationError(f"El fotograma {path.name} no es un PNG válido.")
            try:
                self._proc.stdin.write(data)
            except (BrokenPipeError, OSError) as exc:
                raise VideoCompilationError(
                    "FFmpeg terminó inesperadamente durante la codificación.", detail=self._error_output()
                ) from exc
            path.unlink(missing_ok=True)
            self._next += 1

    def finish(self, expected: int) -> Path:
        """Cierra la entrada, espera a FFmpeg y publica el MP4 de forma atómica.

        Raises:
            VideoCompilationError: secuencia incompleta, error de FFmpeg, timeout o salida vacía.
        """
        assert self._proc is not None and self._proc.stdin is not None
        if self._next != expected:
            raise VideoCompilationError(f"Secuencia incompleta: {self._next} fotogramas de {expected}.")
        try:
            self._proc.stdin.close()
            code = self._proc.wait(timeout=self._timeout)
        except subprocess.TimeoutExpired as exc:
            raise VideoCompilationError(f"FFmpeg superó el tiempo límite de {self._timeout} s.") from exc
        except OSError as exc:
            raise VideoCompilationError("FFmpeg terminó inesperadamente.", detail=self._error_output()) from exc
        if code != 0:
            raise VideoCompilationError(f"FFmpeg terminó con código {code}.", detail=self._error_output())
        if not self._partial.is_file() or self._partial.stat().st_size == 0:
            raise VideoCompilationError("FFmpeg no generó ningún archivo de salida.")
        try:
            os.replace(self._partial, self._output)
        except OSError as exc:
            raise VideoCompilationError("No se pudo guardar el vídeo final.", detail=str(exc)) from exc
        self._finished = True
        return self._output

    def __exit__(self, *_: object) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        if self._proc is not None and self._proc.stdin is not None and not self._proc.stdin.closed:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        self._stderr.close()
        if not self._finished:
            self._partial.unlink(missing_ok=True)

"""Ensamblaje de la secuencia de fotogramas en MP4 mediante FFmpeg."""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
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
    if path is not None:
        return path
    # Fallback persistente: binario en ~/bin (supervive a reinicios del sandbox; se instala vía johnvansickle)
    for cand in (Path.home() / "bin" / "ffmpeg", Path("/home/user/bin/ffmpeg")):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    # Fallback: imageio-ffmpeg (no persistente, solo si está instalado)
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if ff and Path(ff).is_file():
            return ff
    except Exception:
        pass
    raise FFmpegNotFoundError(f"FFmpeg no está disponible ('{binary}' no se encuentra en el PATH).")


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

    Un **hilo vigilante** garantiza que nada se cuelgue para siempre: si FFmpeg
    deja de aceptar datos (disco lleno, proceso atascado), la escritura en la
    tubería bloquearía al worker indefinidamente. Al vencer ``timeout_s`` —o
    ``stall_timeout_s`` sin que FFmpeg haya aceptado un solo fotograma— el
    vigilante detiene la codificación y la escritura bloqueada vuelve con un
    error en lugar de quedarse colgada.

    FFmpeg se lanza en su **propio grupo de procesos** y se detiene con
    ``SIGKILL`` al grupo entero: la escritura solo se desbloquea cuando *ningún*
    proceso conserva abierto el extremo de lectura de la tubería, así que matar
    únicamente al padre no basta si queda un hijo o un envoltorio de shell.

    Uso::

        with StreamingVideoCompiler(out, width=1920, height=1080) as compiler:
            ...                                  # el motor escribe frame_0000.png, frame_0001.png…
            compiler.feed(frames_dir, upto=n)    # consume y borra los fotogramas [siguiente, n)
            compiler.finish(expected=total)

    Si el bloque termina con una excepción, FFmpeg se detiene y el vídeo parcial se elimina.
    """

    #: Cada cuánto despierta el vigilante para revisar los plazos (por defecto).
    _WATCHDOG_TICK_S = 0.5

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
        stall_timeout_s: float = 180.0,
        watchdog_tick_s: float | None = None,
    ) -> None:
        if width % 2 or height % 2:
            raise VideoCompilationError(f"La resolución debe ser par para yuv420p (recibido {width}x{height}).")
        self._output = output_path
        self._partial = output_path.with_name(output_path.name + ".part")
        self._timeout = timeout_s
        #: Máximo sin que FFmpeg acepte un fotograma antes de darlo por atascado.
        self._stall_timeout = stall_timeout_s
        self._tick = watchdog_tick_s if watchdog_tick_s is not None else self._WATCHDOG_TICK_S
        self._next = 0
        #: Total de fotogramas enviados a FFmpeg (los de la ruta más los intercalados).
        self.frames_written = 0
        self._finished = False
        cmd = build_ffmpeg_command(
            find_ffmpeg(ffmpeg_binary), Path("."), self._partial, fps=fps, width=width, height=height, crf=crf, preset=preset
        )
        # Sustituye la entrada por secuencia de archivos por una tubería de PNG
        i = cmd.index("-start_number")
        self._cmd = cmd[:i] + ["-f", "image2pipe", "-c:v", "png", "-i", "pipe:0"] + cmd[i + 4 :]
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr = tempfile.TemporaryFile()  # noqa: SIM115 - se cierra en __exit__
        self._lock = threading.Lock()  # protege el estado que comparten el worker y el vigilante
        self._last_write = 0.0
        self._stopped: str | None = None  # motivo por el que el vigilante mató FFmpeg
        self._finishing = False  # ya no se envían fotogramas: solo se espera el cierre
        self._closed = False  # el compilador ya ha terminado: el vigilante debe salir
        self._watchdog: threading.Thread | None = None

    @property
    def frames_fed(self) -> int:
        """Número de fotogramas entregados a FFmpeg."""
        return self._next

    def __enter__(self) -> StreamingVideoCompiler:
        try:
            self._output.parent.mkdir(parents=True, exist_ok=True)
            self._proc = subprocess.Popen(  # noqa: S603 - lista de argumentos, sin shell
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
                start_new_session=True,  # grupo propio: permite detener también a sus hijos
            )
        except OSError as exc:
            self._stderr.close()
            raise VideoCompilationError("No se pudo iniciar FFmpeg.", detail=str(exc)) from exc
        with self._lock:
            self._last_write = time.monotonic()
        self._watchdog = threading.Thread(
            target=self._watch, name="gpxrender-ffmpeg-watchdog", daemon=True
        )
        self._watchdog.start()
        return self

    def _watch(self) -> None:
        """Mata FFmpeg si se atasca o si la codificación supera el tiempo máximo."""
        started = time.monotonic()
        while True:
            time.sleep(self._tick)
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return  # FFmpeg ya terminó (bien o mal): quien escribe lo notará
            with self._lock:
                if self._closed or self._stopped:
                    return
                now = time.monotonic()
                stalled = now - self._last_write
                elapsed = now - started
                finishing = self._finishing
            reason: str | None = None
            if elapsed > self._timeout and not finishing:
                reason = f"La codificación superó el tiempo límite de {self._timeout} s."
            elif stalled > self._stall_timeout:
                reason = f"FFmpeg dejó de aceptar fotogramas hace {int(stalled)} s (¿disco lleno o proceso atascado?)."
            if reason is None:
                continue
            with self._lock:
                if self._closed or self._stopped:
                    return
                self._stopped = reason
            logger.error("Vigilante: deteniendo FFmpeg. %s", reason)
            self._kill()  # cierra la tubería y desbloquea la escritura en curso
            return

    def _kill(self) -> None:
        """Detiene FFmpeg y todo su grupo de procesos, y espera a que liberen la tubería."""
        proc = self._proc
        if proc is None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # el pid es el del grupo (start_new_session)
        except OSError:  # sin grupo propio, sin permiso o ya desaparecido
            with suppress(OSError):
                proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - solo si el kernel no lo libera
            logger.error("FFmpeg no terminó tras SIGKILL.")

    def _error_output(self) -> str:
        with self._lock:
            if self._stopped:
                return self._stopped
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace").strip()[-2000:]

    def write_png(self, data: bytes) -> None:
        """Envía un fotograma PNG a FFmpeg.

        Puede bloquearse mientras la tubería esté llena; el hilo vigilante mata
        FFmpeg si ese bloqueo se prolonga, lo que convierte la espera en error.

        Raises:
            VideoCompilationError: FFmpeg ha muerto o se ha detenido por plazo.
        """
        assert self._proc is not None and self._proc.stdin is not None, "usar dentro de un bloque 'with'"
        try:
            self._proc.stdin.write(data)
        except (BrokenPipeError, OSError) as exc:
            raise VideoCompilationError(
                "FFmpeg terminó inesperadamente durante la codificación.", detail=self._error_output()
            ) from exc
        with self._lock:  # el fotograma ha sido aceptado: el vigilante reinicia su cuenta
            self._last_write = time.monotonic()
        self.frames_written += 1

    def feed(
        self,
        frames_dir: Path,
        upto: int,
        extra_frames: Callable[[int, bytes], Iterable[bytes]] | None = None,
    ) -> None:
        """Envía a FFmpeg los fotogramas ``[siguiente, upto)`` y los borra del disco.

        Args:
            extra_frames: función ``(índice, png)`` que devuelve fotogramas adicionales a
                insertar justo después de ese fotograma (p. ej. los de un momento).

        Raises:
            VideoCompilationError: falta un fotograma, no es PNG o FFmpeg ha muerto.
        """
        while self._next < upto:
            path = frames_dir / f"frame_{self._next:04d}.png"
            try:
                data = path.read_bytes()
            except OSError as exc:
                raise VideoCompilationError(f"Falta el fotograma {path.name} en la secuencia.") from exc
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise VideoCompilationError(f"El fotograma {path.name} no es un PNG válido.")
            self.write_png(data)
            path.unlink(missing_ok=True)
            if extra_frames is not None:
                for extra in extra_frames(self._next, data):
                    self.write_png(extra)
            self._next += 1

    def finish(self, expected: int) -> Path:
        """Cierra la entrada, espera a FFmpeg y publica el MP4 de forma atómica.

        Raises:
            VideoCompilationError: secuencia incompleta, error de FFmpeg, timeout o salida vacía.
        """
        assert self._proc is not None and self._proc.stdin is not None
        if self._next != expected:
            raise VideoCompilationError(f"Secuencia incompleta: {self._next} fotogramas de {expected}.")
        with self._lock:  # ya no se envían fotogramas: el vigilante pasa a vigilar solo el atasco
            self._finishing = True
        try:
            self._proc.stdin.close()
            code = self._proc.wait(timeout=self._stall_timeout)
        except subprocess.TimeoutExpired as exc:
            self._kill()
            raise VideoCompilationError(
                "FFmpeg no terminó de cerrar el vídeo a tiempo.", detail=self._error_output()
            ) from exc
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
        with self._lock:
            self._finished = True
        return self._output

    def __exit__(self, *_: object) -> None:
        with self._lock:
            self._closed = True  # el vigilante ya no tiene nada que hacer
        if self._watchdog is not None:
            self._watchdog.join(timeout=2.0)
        if self._proc is not None and self._proc.poll() is None:
            self._kill()
        if self._proc is not None and self._proc.stdin is not None and not self._proc.stdin.closed:
            with suppress(OSError):
                self._proc.stdin.close()
        self._stderr.close()
        if not self._finished:
            self._partial.unlink(missing_ok=True)

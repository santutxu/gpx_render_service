"""Jerarquía de excepciones del dominio.

Cada excepción lleva asociado el código HTTP con el que la API debe
responder, de modo que la capa web no necesita conocer los detalles
internos de cada módulo.
"""

from __future__ import annotations


class GPXRenderError(Exception):
    """Excepción base de todo el servicio."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        #: Información técnica adicional (p. ej. stderr de FFmpeg). No se expone al cliente.
        self.detail = detail


class InvalidGPXError(GPXRenderError):
    """El archivo no es un XML/GPX válido o no contiene un track utilizable."""

    status_code = 422
    error_code = "invalid_gpx"


class UnsupportedFileError(GPXRenderError):
    """La extensión o el tipo del archivo subido no es admisible."""

    status_code = 415
    error_code = "unsupported_file"


class FileTooLargeError(GPXRenderError):
    """El archivo supera el tamaño máximo permitido."""

    status_code = 413
    error_code = "file_too_large"


class RenderEngineError(GPXRenderError):
    """El motor gráfico headless falló o produjo una secuencia de fotogramas inválida."""

    error_code = "render_engine_error"


class VideoCompilationError(GPXRenderError):
    """FFmpeg no pudo ensamblar el vídeo."""

    error_code = "video_compilation_error"


class FFmpegNotFoundError(VideoCompilationError):
    """El binario de FFmpeg no está instalado o no es accesible en el PATH."""

    status_code = 503
    error_code = "ffmpeg_not_found"

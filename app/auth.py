"""Control de acceso: claves de API, sesiones de navegador y propiedad de los recursos.

Modelo:
  * Cada cliente se identifica con una **clave de API** asociada a un nombre de
    usuario (``GPX_API_KEYS="ana:clave1,bob:clave2,root:clave3:admin"``).
  * La clave se envía como ``Authorization: Bearer <clave>`` o ``X-API-Key: <clave>``.
  * El navegador (galería, etiqueta ``<video>``) no puede añadir cabeceras, así
    que ``POST /api/v1/auth/login`` canjea la clave por una cookie de sesión
    firmada (HMAC-SHA256), ``HttpOnly`` y ``SameSite=Strict``.
  * Cada trabajo y cada vídeo pertenece a quien lo creó. Los demás usuarios
    reciben 404 (no se revela su existencia); los administradores lo ven todo.

Si no se configura ninguna clave, se genera una de administrador y se guarda
en ``<GPX_OUTPUT_DIR>/.api_key`` (permisos 600): el servicio nunca queda abierto
por descuido. Para desactivar la autenticación hay que pedirlo expresamente con
``GPX_AUTH_DISABLED=1``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings
from .exceptions import GPXRenderError

logger = logging.getLogger(__name__)

SESSION_COOKIE = "gpx_session"
MIN_KEY_LENGTH = 16


class AuthenticationError(GPXRenderError):
    """Faltan credenciales o no son válidas."""

    status_code = 401
    error_code = "unauthorized"


class AuthConfigurationError(RuntimeError):
    """La configuración de claves es inválida (se detecta al arrancar)."""


@dataclass(frozen=True)
class Principal:
    """Usuario autenticado."""

    name: str
    is_admin: bool = False

    def can_access(self, owner: str | None) -> bool:
        """¿Puede este usuario acceder a un recurso del propietario indicado?

        Los recursos sin propietario (creados antes de activar el control de
        acceso) solo son visibles para administradores.
        """
        return self.is_admin or (owner is not None and owner == self.name)


def _digest(value: str) -> bytes:
    return hashlib.sha256(value.encode()).digest()


class AuthManager:
    """Valida claves de API y emite/verifica sesiones firmadas."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.enabled = not settings.auth_disabled
        self.generated_key_path: str | None = None
        # Solo se conservan los hashes de las claves, nunca el texto en claro.
        self._keys: dict[bytes, Principal] = {}
        if not self.enabled:
            logger.warning("Autenticación DESACTIVADA (GPX_AUTH_DISABLED): cualquiera puede usar la API.")
            self._secret = secrets.token_bytes(32)
            return

        raw = settings.api_keys.strip()
        if raw:
            self._load_keys(raw)
        else:
            self._load_or_generate_key()
        material = settings.session_secret or "|".join(sorted(k.hex() for k in self._keys))
        self._secret = _digest("gpx-session:" + material)  # cambiar las claves invalida las sesiones

    # ------------------------------------------------------------------ #
    def _load_keys(self, raw: str) -> None:
        names: set[str] = set()
        for entry in filter(None, (e.strip() for e in raw.split(","))):
            parts = entry.split(":")
            if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "admin"):
                raise AuthConfigurationError("GPX_API_KEYS: cada entrada debe ser 'usuario:clave' o 'usuario:clave:admin'.")
            name, key = parts[0].strip(), parts[1].strip()
            if not name or not name.replace("-", "").replace("_", "").isalnum():
                raise AuthConfigurationError(f"GPX_API_KEYS: nombre de usuario inválido ('{name}').")
            if name in names:
                raise AuthConfigurationError(f"GPX_API_KEYS: usuario duplicado ('{name}').")
            if len(key) < MIN_KEY_LENGTH:
                raise AuthConfigurationError(
                    f"GPX_API_KEYS: la clave de '{name}' debe tener al menos {MIN_KEY_LENGTH} caracteres."
                )
            names.add(name)
            self._keys[_digest(key)] = Principal(name=name, is_admin=len(parts) == 3)

    def _load_or_generate_key(self) -> None:
        path = self._settings.output_dir / ".api_key"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file() and len(key := path.read_text(encoding="utf-8").strip()) >= MIN_KEY_LENGTH:
                logger.info("Clave de API de administrador leída de %s", path)
            else:
                key = secrets.token_urlsafe(32)
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(key + "\n")
                logger.warning("GPX_API_KEYS no definida. Clave de administrador generada y guardada en %s", path)
        except OSError as exc:
            raise AuthConfigurationError(f"No se pudo leer/crear la clave de API en {path}: {exc}") from exc
        self.generated_key_path = str(path)
        self._keys[_digest(key)] = Principal(name="admin", is_admin=True)

    # ------------------------------------------------------------------ #
    def authenticate_key(self, key: str) -> Principal | None:
        """Devuelve el usuario de una clave de API, o ``None`` si no es válida."""
        candidate = _digest(key)
        found: Principal | None = None
        for stored, principal in self._keys.items():  # comparación en tiempo constante, sin salida anticipada
            if hmac.compare_digest(stored, candidate):
                found = principal
        return found

    def _sign(self, body: str) -> str:
        return base64.urlsafe_b64encode(hmac.new(self._secret, body.encode(), hashlib.sha256).digest()).decode().rstrip("=")

    def create_session(self, principal: Principal) -> str:
        """Token de sesión firmado: ``usuario.caducidad.firma``."""
        body = f"{principal.name}.{int(time.time()) + self._settings.session_ttl_s}"
        return f"{body}.{self._sign(body)}"

    def verify_session(self, token: str) -> Principal | None:
        """Valida firma y caducidad, y que el usuario siga existiendo (permite revocar)."""
        try:
            name, expires, signature = token.split(".")
            if not hmac.compare_digest(signature, self._sign(f"{name}.{expires}")) or int(expires) < time.time():
                return None
        except ValueError:
            return None
        return next((p for p in self._keys.values() if p.name == name), None)


_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    """Gestor de autenticación (singleton perezoso)."""
    global _manager  # noqa: PLW0603
    if _manager is None:
        _manager = AuthManager(get_settings())
    return _manager


# auto_error=False: la respuesta 401 la genera require_user con un formato homogéneo.
_bearer = HTTPBearer(auto_error=False, description="Clave de API")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Clave de API")


async def require_user(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(_bearer),
    header_key: str | None = Depends(_api_key_header),
) -> Principal:
    """Dependencia de FastAPI: exige un usuario autenticado.

    Orden de comprobación: ``Authorization: Bearer`` → ``X-API-Key`` → cookie de sesión.

    Raises:
        AuthenticationError: 401 si no hay credenciales válidas.
    """
    manager = get_auth_manager()
    if not manager.enabled:
        return Principal(name="anonymous", is_admin=True)

    key = bearer.credentials if bearer else header_key
    if key:
        principal = manager.authenticate_key(key)
        if principal is None:
            raise AuthenticationError("Clave de API no válida.")
        return principal

    token = request.cookies.get(SESSION_COOKIE)
    if token and (principal := manager.verify_session(token)) is not None:
        return principal
    raise AuthenticationError("Se requiere autenticación: envía 'Authorization: Bearer <clave>' o 'X-API-Key'.")

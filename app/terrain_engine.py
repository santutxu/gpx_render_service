"""Motor 3D headless: relieve real texturizado con imagen satélite (OpenGL vía ModernGL).

Renderiza fuera de pantalla (EGL), por lo que no necesita servidor gráfico.
Usa la GPU si existe y, si no, el rasterizador por software de Mesa (llvmpipe).

Escena por fotograma:
  1. cielo en degradado;
  2. malla del terreno con ortofoto, sombreado según la pendiente y bruma de distancia;
  3. recorrido completo (blanco translúcido) y tramo ya recorrido (naranja), ceñidos al relieve;
  4. marcador de posición, siempre visible;
  5. HUD 2D (distancia, altitud, tiempo, perfil altimétrico y atribución del mapa).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

from .config import Settings
from .exceptions import GPXRenderError, RenderEngineError
from .schemas import CameraStyle, RenderPayload
from .terrain import TerrainData, build_terrain

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]

_SKY_VS = """
#version 330
out vec2 v_pos;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
    v_pos = p;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""
_SKY_FS = """
#version 330
in vec2 v_pos;
out vec4 f_color;
uniform vec3 u_haze;
void main() {
    float t = clamp(v_pos.y * 0.5 + 0.5, 0.0, 1.0);
    f_color = vec4(mix(u_haze, vec3(0.30, 0.52, 0.85), smoothstep(0.35, 1.0, t)), 1.0);
}
"""
_TERRAIN_VS = """
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_world;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    v_world = in_pos;
    v_normal = in_normal;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""
_TERRAIN_FS = """
#version 330
uniform sampler2D u_tex;
uniform vec3 u_eye;
uniform vec3 u_sun;
uniform vec3 u_haze;
uniform vec2 u_fog;   // inicio, fin (m)
in vec3 v_world;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec3 base = texture(u_tex, v_uv).rgb;
    float diffuse = max(dot(normalize(v_normal), u_sun), 0.0);
    vec3 color = base * (0.50 + 0.75 * diffuse);          // sombreado: resalta el relieve
    float grey = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(grey), color, 1.15);                  // algo más de saturación
    float fog = smoothstep(u_fog.x, u_fog.y, distance(u_eye, v_world)) * 0.9;
    f_color = vec4(mix(color, u_haze, fog), 1.0);
}
"""
_LINE_VS = """
#version 330
uniform mat4 u_mvp;
uniform vec2 u_viewport;
uniform float u_width;
in vec3 in_a;
in vec3 in_b;
in float in_t;
in float in_side;
in float in_idx;
out float v_idx;
void main() {
    v_idx = in_idx + in_t;
    vec4 a = u_mvp * vec4(in_a, 1.0);
    vec4 b = u_mvp * vec4(in_b, 1.0);
    const float eps = 1.0;
    if (a.w < eps && b.w < eps) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    if (a.w < eps) a = mix(a, b, (eps - a.w) / (b.w - a.w));   // recorte manual contra el plano cercano
    if (b.w < eps) b = mix(b, a, (eps - b.w) / (a.w - b.w));
    vec2 half_vp = u_viewport * 0.5;
    vec2 d = b.xy / b.w * half_vp - a.xy / a.w * half_vp;
    float len = length(d);
    d = len > 1e-5 ? d / len : vec2(1.0, 0.0);
    vec2 n = vec2(-d.y, d.x);
    vec4 p = mix(a, b, in_t);
    vec2 offset = n * in_side * u_width * 0.5 + d * (in_t * 2.0 - 1.0) * u_width * 0.25;
    p.xy += offset / half_vp * p.w;                            // grosor constante en píxeles
    gl_Position = p;
}
"""
_LINE_FS = """
#version 330
uniform vec4 u_color;
uniform float u_progress;
in float v_idx;
out vec4 f_color;
void main() {
    if (v_idx > u_progress) discard;
    f_color = u_color;
}
"""
_MARKER_VS = """
#version 330
uniform mat4 u_mvp;
uniform vec3 u_center;
uniform vec2 u_viewport;
uniform float u_radius;
in vec2 in_corner;
out vec2 v_corner;
void main() {
    v_corner = in_corner;
    vec4 c = u_mvp * vec4(u_center, 1.0);
    c.xy += in_corner * u_radius / (u_viewport * 0.5) * c.w;
    gl_Position = c;
}
"""
_MARKER_FS = """
#version 330
in vec2 v_corner;
out vec4 f_color;
void main() {
    float r = length(v_corner);
    if (r > 1.0) discard;
    vec3 color = r > 0.62 ? vec3(1.0) : vec3(1.0, 0.36, 0.12);
    f_color = vec4(color, 1.0 - smoothstep(0.92, 1.0, r));
}
"""

_ORANGE = (1.0, 0.36, 0.12, 1.0)


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> FloatArray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )


def _look_at(eye: FloatArray, target: FloatArray) -> FloatArray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    norm = np.linalg.norm(right)
    right = right / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])  # vista exactamente cenital
    up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view


class Terrain3DEngine:
    """Motor 3D real: terreno con relieve + imagen satélite + recorrido animado."""

    name = "terrain3d"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ------------------------------------------------------------------ #
    def render(self, payload: RenderPayload, payload_path: Path, frames_dir: Path, on_progress: Any | None) -> None:
        try:
            import moderngl
        except ImportError as exc:  # dependencia opcional del motor
            raise RenderEngineError("El motor 3D requiere el paquete 'moderngl'.") from exc

        frames = payload.frames
        n = len(frames)
        fx = np.fromiter((f.x for f in frames), np.float64, n)
        fy = np.fromiter((f.y for f in frames), np.float64, n)
        cfg = payload.camera
        exag = cfg.vertical_exaggeration

        extent = float(max(np.ptp(fx), np.ptp(fy)))
        margin = max(
            self._settings.terrain_margin_ratio * extent, self._settings.terrain_min_margin_m, 1.5 * cfg.distance_m
        )
        terrain = build_terrain(fx, fy, payload.origin["lat"], payload.origin["lon"], margin, self._settings)

        ground = terrain.height_at(fx, fy)
        targets = np.column_stack((fx, fy, ground * exag))
        eyes = self._camera_path(payload, terrain, targets)

        ctx = self._create_context(moderngl)
        try:
            self._render_frames(moderngl, ctx, payload, terrain, targets, eyes, ground, frames_dir, on_progress)
        except GPXRenderError:  # incluye errores del consumidor de fotogramas (FFmpeg)
            raise
        except Exception as exc:  # noqa: BLE001 - errores de driver/shader
            raise RenderEngineError("Fallo del motor OpenGL durante el render.", detail=repr(exc)) from exc
        finally:
            ctx.release()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _create_context(moderngl: Any) -> Any:
        errors: list[str] = []
        for kwargs in ({"backend": "egl"}, {}):
            try:
                ctx = moderngl.create_context(standalone=True, require=330, **kwargs)
                logger.info("OpenGL: %s (%s)", ctx.info.get("GL_RENDERER"), ctx.info.get("GL_VERSION"))
                return ctx
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))
        raise RenderEngineError(
            "No se pudo crear un contexto OpenGL headless. Instala los drivers EGL/Mesa "
            "(Debian/Ubuntu: 'sudo apt install libegl1 libgl1-mesa-dri').",
            detail=" | ".join(errors),
        )

    def _camera_path(self, payload: RenderPayload, terrain: TerrainData, targets: FloatArray) -> FloatArray:
        """Posición de la cámara por fotograma: según estilo, sin atravesar montañas y suavizada."""
        cfg = payload.camera
        n = len(payload.frames)
        heading = np.fromiter((f.heading for f in payload.frames), np.float64, n)
        t = np.fromiter((f.t for f in payload.frames), np.float64, n)
        if cfg.style is CameraStyle.FOLLOW:
            azimuth = np.deg2rad(heading)
        elif cfg.style is CameraStyle.ORBIT:
            azimuth = np.deg2rad(heading[0] + cfg.orbit_deg_per_s * t)
        else:
            azimuth = np.zeros(n)
        pitch = math.radians(cfg.pitch_deg)
        horizontal = cfg.distance_m * math.cos(pitch)
        eyes = np.column_stack(
            (
                targets[:, 0] - horizontal * np.sin(azimuth),
                targets[:, 1] - horizontal * np.cos(azimuth),
                targets[:, 2] + cfg.distance_m * math.sin(pitch),
            )
        )
        sigma = payload.fps * 1.0
        eyes = gaussian_filter1d(eyes, sigma=sigma, axis=0, mode="nearest")
        # Altura mínima sobre el terreno bajo la cámara (suavizada para que no dé tirones)
        clearance = max(0.12 * cfg.distance_m, 60.0)
        floor = terrain.height_at(eyes[:, 0], eyes[:, 1]) * cfg.vertical_exaggeration + clearance
        lift = gaussian_filter1d(np.maximum(floor - eyes[:, 2], 0.0), sigma=sigma, mode="nearest")
        eyes[:, 2] = np.maximum(eyes[:, 2] + lift * 1.5, floor)
        return eyes

    @staticmethod
    def _terrain_buffers(terrain: TerrainData, exag: float) -> tuple[bytes, bytes]:
        rows, cols = terrain.z.shape
        z = terrain.z * exag
        dzdx = np.gradient(z, axis=1) / np.gradient(terrain.x, axis=1)
        dzdy = np.gradient(z, axis=0) / np.gradient(terrain.y, axis=0)
        normal = np.dstack((-dzdx, -dzdy, np.ones_like(z)))
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        uu, vv = np.meshgrid(terrain.u, terrain.v)
        vertices = np.dstack((terrain.x, terrain.y, z, normal, uu, vv)).astype("f4")
        idx = np.arange(rows * cols, dtype=np.uint32).reshape(rows, cols)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        indices = np.stack((a, c, b, b, c, d), axis=-1)
        return vertices.tobytes(), indices.astype("u4").tobytes()

    @staticmethod
    def _line_buffer(points: FloatArray) -> bytes:
        """Cada segmento se expande a un quad (2 triángulos) orientado a cámara en el vertex shader."""
        segs = len(points) - 1
        corners = np.array([[0, -1], [1, -1], [0, 1], [0, 1], [1, -1], [1, 1]], dtype="f4")  # (t, lado)
        data = np.empty((segs, 6, 9), dtype="f4")
        data[:, :, 0:3] = points[:-1, None, :]
        data[:, :, 3:6] = points[1:, None, :]
        data[:, :, 6:8] = corners[None, :, :]
        data[:, :, 8] = np.arange(segs, dtype="f4")[:, None]
        return data.tobytes()

    # ------------------------------------------------------------------ #
    def _render_frames(
        self,
        moderngl: Any,
        ctx: Any,
        payload: RenderPayload,
        terrain: TerrainData,
        targets: FloatArray,
        eyes: FloatArray,
        ground: FloatArray,
        frames_dir: Path,
        on_progress: Any | None,
    ) -> None:
        w, h = payload.width, payload.height
        cfg = payload.camera
        n = len(payload.frames)
        haze = (0.78, 0.85, 0.93)
        sun = np.array([-0.55, 0.45, 0.70])
        sun /= np.linalg.norm(sun)

        # --- Framebuffers (MSAA ×4 → resolución) -----------------------------
        try:
            scene = ctx.framebuffer([ctx.renderbuffer((w, h), 3, samples=4)], ctx.depth_renderbuffer((w, h), samples=4))
        except Exception:  # noqa: BLE001 - driver sin multisampling
            scene = ctx.framebuffer([ctx.renderbuffer((w, h), 3)], ctx.depth_renderbuffer((w, h)))
        resolved = ctx.framebuffer([ctx.renderbuffer((w, h), 3)])

        # --- Programas y geometría -------------------------------------------
        sky = ctx.program(vertex_shader=_SKY_VS, fragment_shader=_SKY_FS)
        sky["u_haze"].value = haze
        sky_vao = ctx.vertex_array(sky, [])

        prog = ctx.program(vertex_shader=_TERRAIN_VS, fragment_shader=_TERRAIN_FS)
        vertices, indices = self._terrain_buffers(terrain, cfg.vertical_exaggeration)
        terrain_vao = ctx.vertex_array(
            prog,
            [(ctx.buffer(vertices), "3f 3f 2f", "in_pos", "in_normal", "in_uv")],
            index_buffer=ctx.buffer(indices),
            index_element_size=4,
        )
        th, tw = terrain.texture.shape[:2]
        limit = ctx.info.get("GL_MAX_TEXTURE_SIZE", 4096)
        image = terrain.texture
        if max(tw, th) > limit:
            image = np.asarray(Image.fromarray(image).resize((min(tw, limit), min(th, limit))))
            th, tw = image.shape[:2]
        texture = ctx.texture((tw, th), 3, np.ascontiguousarray(image).tobytes(), alignment=1)
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        texture.anisotropy = 8.0
        texture.repeat_x = texture.repeat_y = False
        texture.use(0)
        prog["u_tex"].value = 0
        prog["u_sun"].value = tuple(sun)
        prog["u_haze"].value = haze
        diag = float(math.hypot(np.ptp(terrain.x), np.ptp(terrain.y)))
        prog["u_fog"].value = (3.0 * cfg.distance_m, max(diag * 0.9, 9.0 * cfg.distance_m))

        line = ctx.program(vertex_shader=_LINE_VS, fragment_shader=_LINE_FS)
        track_pts = targets.copy()
        track_pts[:, 2] += 4.0 + 0.01 * cfg.distance_m  # ligeramente sobre el suelo para evitar z-fighting
        line_vao = ctx.vertex_array(
            line, [(ctx.buffer(self._line_buffer(track_pts)), "3f 3f 1f 1f 1f", "in_a", "in_b", "in_t", "in_side", "in_idx")]
        )
        line["u_viewport"].value = (w, h)
        line_px = max(h / 180.0, 3.0)

        marker = ctx.program(vertex_shader=_MARKER_VS, fragment_shader=_MARKER_FS)
        marker_vao = ctx.vertex_array(
            marker, [(ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes()), "2f", "in_corner")]
        )
        marker["u_viewport"].value = (w, h)
        marker["u_radius"].value = max(h / 60.0, 8.0)

        hud = _Hud(payload, ground, terrain.attribution)
        z_top = float(terrain.z.max() * cfg.vertical_exaggeration)

        ctx.enable(moderngl.BLEND)
        for i in range(n):
            eye, target = eyes[i], targets[i]
            dist = float(np.linalg.norm(target - eye))
            near = max(dist * 0.02, 2.0)
            far = dist + diag + z_top + 1000.0
            mvp = (_perspective(cfg.fov_deg, w / h, near, far) @ _look_at(eye.copy(), target)).T.astype("f4").tobytes()

            scene.use()
            scene.depth_mask = True
            scene.clear(*haze, 1.0, depth=1.0)
            ctx.disable(moderngl.DEPTH_TEST)
            sky_vao.render(moderngl.TRIANGLES, vertices=3)

            ctx.enable(moderngl.DEPTH_TEST)
            prog["u_mvp"].write(mvp)
            prog["u_eye"].value = tuple(eye)
            terrain_vao.render(moderngl.TRIANGLES)

            scene.depth_mask = False  # las líneas se ocultan tras las montañas pero no compiten entre sí
            line["u_mvp"].write(mvp)
            for color, width, progress in (
                ((1.0, 1.0, 1.0, 0.55), line_px * 0.7, 1e9),  # recorrido pendiente
                ((0.10, 0.05, 0.02, 0.65), line_px + 3.0, float(i)),  # contorno del tramo hecho
                (_ORANGE, line_px, float(i)),
            ):
                line["u_color"].value = color
                line["u_width"].value = width
                line["u_progress"].value = progress
                line_vao.render(moderngl.TRIANGLES)

            ctx.disable(moderngl.DEPTH_TEST)  # el marcador se ve siempre
            marker["u_mvp"].write(mvp)
            marker["u_center"].value = tuple(track_pts[i])
            marker_vao.render(moderngl.TRIANGLE_STRIP)

            ctx.copy_framebuffer(resolved, scene)
            frame = Image.frombytes("RGB", (w, h), resolved.read(components=3, alignment=1))
            frame = frame.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            hud.draw(frame, i)
            try:
                frame.save(frames_dir / f"frame_{i:04d}.png", format="PNG", compress_level=1)
            except OSError as exc:
                raise RenderEngineError("Error de E/S al escribir los fotogramas.", detail=str(exc)) from exc
            if on_progress and (i % 10 == 0 or i == n - 1):
                on_progress(i + 1, n)


class _Hud:
    """Capa 2D superpuesta: datos en vivo, perfil altimétrico y atribución."""

    def __init__(self, payload: RenderPayload, ground: FloatArray, attribution: str) -> None:
        self._p = payload
        w, h = payload.width, payload.height
        self._font = ImageFont.load_default(size=max(h // 30, 14))
        self._small = ImageFont.load_default(size=max(h // 60, 10))
        self._attribution = attribution
        n = len(payload.frames)
        # Altitud mostrada: la del GPX si existe; si no, la del modelo del terreno
        if payload.metrics.has_elevation:
            self._ele = np.fromiter((f.ele for f in payload.frames), np.float64, n)
        else:
            self._ele = ground
        dist = np.fromiter((f.dist for f in payload.frames), np.float64, n)
        self._box = (int(w * 0.03), int(h * 0.80), int(w * 0.97), int(h * 0.955))
        x0, y0, x1, y1 = self._box
        pad = h * 0.015
        self._idx = np.unique(np.linspace(0, n - 1, min(n, 400)).astype(int))  # perfil decimado
        span = max(float(np.ptp(self._ele)), 1.0)
        self._px = x0 + pad + dist[self._idx] / max(dist[-1], 1.0) * (x1 - x0 - 2 * pad)
        self._py = y1 - pad - (self._ele[self._idx] - self._ele.min()) / span * (y1 - y0 - 2.4 * pad)
        self._base = y1 - pad * 0.6

    def draw(self, image: Image.Image, i: int) -> None:
        frame = self._p.frames[i]
        w, h = image.size
        d = ImageDraw.Draw(image, "RGBA")
        radius = h // 60

        secs = int(frame.elapsed)
        text = f"{frame.dist / 1000:.2f} km    {self._ele[i]:.0f} m    {secs // 3600}:{secs % 3600 // 60:02d}:{secs % 60:02d}"
        tx, ty = int(w * 0.03), int(h * 0.04)
        bbox = d.textbbox((tx, ty), text, font=self._font)
        d.rounded_rectangle((bbox[0] - radius, bbox[1] - radius, bbox[2] + radius, bbox[3] + radius), radius, fill=(0, 0, 0, 120))
        d.text((tx, ty), text, font=self._font, fill=(255, 255, 255, 255))

        d.rounded_rectangle(self._box, radius, fill=(0, 0, 0, 110))
        pts = list(zip(self._px.tolist(), self._py.tolist()))
        d.polygon([(pts[0][0], self._base), *pts, (pts[-1][0], self._base)], fill=(255, 255, 255, 70))
        k = int(np.searchsorted(self._idx, i, side="right"))
        if k >= 2:
            done = pts[:k]
            d.polygon([(done[0][0], self._base), *done, (done[-1][0], self._base)], fill=(255, 92, 30, 230))
        d.line(pts, fill=(255, 255, 255, 200), width=max(h // 400, 1))

        if self._attribution:
            bbox = d.textbbox((0, 0), self._attribution, font=self._small)
            d.text((w - bbox[2] - h * 0.015, h - bbox[3] - h * 0.012), self._attribution, font=self._small, fill=(255, 255, 255, 190))

# GPX 3D Render Service

Backend FastAPI que convierte un `.gpx` en un vídeo MP4 animado (estilo Relive).

```
POST /api/v1/render-gpx ─▶ gpx_processor ─▶ renderer_orchestrator ─▶ video_compiler ─▶ MP4
   (valida, 202 + job)     (limpia/suaviza)   (payload JSON → PNGs)     (FFmpeg/libx264)
```

## Estructura

| Archivo | Responsabilidad |
|---|---|
| `app/main.py` | API FastAPI, validación de subida, background tasks |
| `app/gpx_processor.py` | Parsing, métricas, filtro anti-saltos, suavizado, remuestreo por fotograma |
| `app/renderer_orchestrator.py` | Payload JSON, motores `simulated` / `node`, verificación de fotogramas |
| `app/terrain.py`, `app/terrain_engine.py` | Descarga de relieve + satélite y render 3D con OpenGL headless |
| `app/video_compiler.py` | `subprocess` + FFmpeg (libx264, yuv420p, 30 FPS), por lotes o en streaming, limpieza |
| `app/jobs.py` | Registro de trabajos y pipeline en segundo plano con directorio temporal |
| `app/schemas.py`, `config.py`, `exceptions.py` | Modelos Pydantic, configuración, errores de dominio |

## Uso

```bash
pip install -r requirements.txt          # + FFmpeg en el PATH
uvicorn app.main:app --host 0.0.0.0 --port 8000

curl -F file=@ruta.gpx -F resolution=720p -F camera_style=orbit -F animation_speed=1.5 \
     http://localhost:8000/api/v1/render-gpx        # → 202 {job_id, metrics, status_url}
curl http://localhost:8000/api/v1/jobs/<job_id>      # estado y progreso
curl -O http://localhost:8000/api/v1/jobs/<job_id>/video
```

Parámetros: `animation_speed` 0.5–4.0, `resolution` `720p|1080p`, `camera_style` `follow|orbit|overhead`.
Otros endpoints: `GET /api/v1/jobs/{id}/track` (puntos interpolados), `GET /health`, `/docs`.

Pruebas y demo sin servidor:

```bash
pip install -r requirements-dev.txt && pytest -q
python demo.py [ruta.gpx] [follow|orbit|overhead] [720p|1080p]   # genera demo_output.mp4
```

## Motor gráfico

`GPX_RENDER_ENGINE` selecciona el motor:

| Valor | Qué hace | Requisitos |
|---|---|---|
| `terrain3d` (por defecto) | 3D real: relieve de las montañas + imagen satélite, recorrido ceñido al terreno, cámara animada, HUD con perfil | `moderngl`, OpenGL headless (EGL), conexión a internet |
| `simulated` | Mapa 2D plano con Pillow (pruebas, sin red) | ninguno |
| `node` | Lanza `node $GPX_NODE_WORKER_SCRIPT --payload payload.json --out frames/` (worker no incluido) | Node.js |

Motor `terrain3d` (`app/terrain.py` + `app/terrain_engine.py`):

- **Relieve**: teselas *terrarium* de Mapzen/AWS. **Imagen**: Esri World Imagery (la atribución se
  pinta en el vídeo; revisa sus condiciones de uso antes de un uso comercial). Ambas URL son
  configurables (`GPX_DEM_URL`, `GPX_IMAGERY_URL`, `GPX_IMAGERY_ATTRIBUTION`).
- Las teselas se cachean en `GPX_TILE_CACHE_DIR`; la primera vez que se renderiza una zona tarda más.
- Usa la GPU si hay; si no, el rasterizador por software de Mesa (mucho más lento: ~0,5 s por
  fotograma a 1080p). En Debian/Ubuntu: `sudo apt install libegl1 libgl1-mesa-dri`.
- Los fotogramas se envían a FFmpeg y se borran según se generan (`StreamingVideoCompiler`), así
  que el directorio temporal no crece aunque el vídeo sea largo.
- Calidad: `GPX_IMAGERY_MAX_PX` (lado máximo de la textura, 6144 por defecto) y
  `GPX_TERRAIN_MESH_MAX` (vértices por lado de la malla, 480).

## Configuración (variables de entorno)

`GPX_MAX_UPLOAD_BYTES`, `GPX_SECONDS_PER_KM`, `GPX_MIN_VIDEO_SECONDS`, `GPX_MAX_VIDEO_SECONDS`,
`GPX_RENDER_ENGINE`, `GPX_NODE_WORKER_SCRIPT`, `GPX_MAX_CONCURRENT_RENDERS`, `GPX_FFMPEG_BINARY`,
`GPX_FFMPEG_CRF`, `GPX_FFMPEG_PRESET`, `GPX_OUTPUT_DIR` (ver `app/config.py`).

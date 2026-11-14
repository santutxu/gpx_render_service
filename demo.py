"""Demo sin servidor: ejecuta el pipeline completo sobre un GPX.

Uso: python demo.py [ruta.gpx] [follow|orbit|overhead] [720p|1080p]
"""
import sys, tempfile, time
from pathlib import Path

from app.config import get_settings
from app.gpx_processor import process_gpx
from app.renderer_orchestrator import RendererOrchestrator
from app.schemas import CameraStyle, RenderOptions, Resolution
from app.video_compiler import StreamingVideoCompiler

args = sys.argv[1:]
data = Path(args[0] if args else "sample_mountain.gpx").read_bytes()
opts = RenderOptions(
    camera_style=CameraStyle(args[1]) if len(args) > 1 else CameraStyle.FOLLOW,
    resolution=Resolution(args[2]) if len(args) > 2 else Resolution.P1080,
)
s = get_settings()
t = time.time()
track = process_gpx(data, opts, s)
print(track.metrics.model_dump_json(indent=2))
orch = RendererOrchestrator(settings=s)
payload = orch.build_payload("demo", track, opts)
out = Path("demo_output.mp4")
with tempfile.TemporaryDirectory(prefix="gpxrender_demo_") as tmp, StreamingVideoCompiler(
    out, width=payload.width, height=payload.height, fps=payload.fps
) as compiler:
    frames_dir = Path(tmp) / "frames"

    def on_progress(done: int, total: int) -> None:
        compiler.feed(frames_dir, done)
        print(f"\r{done}/{total}", end="", flush=True)

    orch.render_frames(payload, Path(tmp), on_progress, verify=False)
    compiler.feed(frames_dir, payload.frame_count)
    compiler.finish(payload.frame_count)
print(f"\n{payload.frame_count} fotogramas -> {out} ({out.stat().st_size / 1e6:.1f} MB) en {time.time() - t:.1f}s")

"""Pruebas de extremo a extremo y de casos límite. Ejecutar con ``pytest -q``."""

from __future__ import annotations

import math
import os
import subprocess
from datetime import datetime, timedelta, timezone

os.environ.setdefault("GPX_MIN_VIDEO_SECONDS", "2")  # vídeos cortos para que las pruebas sean rápidas
os.environ.setdefault("GPX_FFMPEG_PRESET", "ultrafast")
os.environ.setdefault("GPX_RENDER_ENGINE", "simulated")  # sin red ni OpenGL en las pruebas

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.exceptions import InvalidGPXError, VideoCompilationError
from app.gpx_processor import process_gpx
from app.main import app
from app.schemas import RenderOptions
from app.video_compiler import compile_video


def make_gpx(n: int = 300, *, with_time: bool = True, with_ele: bool = True, spike_at: int | None = None) -> bytes:
    """Track sintético: ~1,6 km hacia el nordeste desde Bilbao con ruido GPS y una colina."""
    rng = np.random.default_rng(42)
    t0 = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)
    pts = []
    for i in range(n):
        lat = 43.2630 + i * 4e-5 + rng.normal(0, 1.5e-5)
        lon = -2.9350 + i * 4e-5 + rng.normal(0, 1.5e-5)
        if i == spike_at:
            lat += 0.01  # salto de ~1,1 km
        ele = f"<ele>{50 + 100 * math.sin(math.pi * i / n) + rng.normal(0, 1.0):.1f}</ele>" if with_ele else ""
        tm = f"<time>{(t0 + timedelta(seconds=5 * i)).isoformat()}</time>" if with_time else ""
        pts.append(f'<trkpt lat="{lat:.7f}" lon="{lon:.7f}">{ele}{tm}</trkpt>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">'
        f"<trk><name>t</name><trkseg>{''.join(pts)}</trkseg></trk></gpx>"
    ).encode()


OPTS = RenderOptions()


def test_metrics_are_plausible() -> None:
    track = process_gpx(make_gpx(), OPTS)
    m = track.metrics
    assert 1500 < m.total_distance_m < 2600
    assert 80 < m.elevation_gain_m < 125 and 80 < m.elevation_loss_m < 125
    assert m.duration_s == pytest.approx(299 * 5)
    assert not m.duration_is_estimated and m.avg_speed_kmh > 0
    assert track.frame_count == round(track.video_duration_s * 30)
    assert np.all(np.diff(track.distance_m) >= 0) and np.all(np.diff(track.video_t) > 0)


def test_spike_is_removed_and_path_is_smooth() -> None:
    clean = process_gpx(make_gpx(), OPTS)
    spiky = process_gpx(make_gpx(spike_at=150), OPTS)
    assert spiky.metrics.outliers_corrected >= 1
    assert spiky.metrics.total_distance_m == pytest.approx(clean.metrics.total_distance_m, rel=0.05)
    assert spiky.lat.max() < 43.2630 + 300 * 4e-5 + 0.001


def test_without_time_or_elevation() -> None:
    m = process_gpx(make_gpx(with_time=False, with_ele=False), OPTS).metrics
    assert m.duration_is_estimated and not m.has_timestamps and not m.has_elevation
    assert m.elevation_gain_m == 0 and m.duration_s > 0


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"esto no es xml",
        b"<gpx><trk><trkseg><trkpt lat='1' lon='1'>",  # truncado
        b'<?xml version="1.0"?><kml></kml>',
        b'<?xml version="1.0"?><gpx version="1.1"></gpx>',  # sin puntos
        b'<?xml version="1.0"?><gpx version="1.1"><trk><trkseg><trkpt lat="x" lon="y"/></trkseg></trk></gpx>',
        b'<?xml version="1.0"?><!DOCTYPE g [<!ENTITY a "aaaa">]><gpx version="1.1">&a;</gpx>',
        b"\x89PNG\r\n\x1a\n\x00\x00\x00",
    ],
)
def test_invalid_gpx_rejected(payload: bytes) -> None:
    with pytest.raises(InvalidGPXError):
        process_gpx(payload, OPTS)


def test_compile_rejects_gaps(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "frame_0000.png").write_bytes(b"x")
    (tmp_path / "frame_0002.png").write_bytes(b"x")
    with pytest.raises(VideoCompilationError):
        compile_video(tmp_path, tmp_path / "o.mp4", width=1280, height=720)
    assert (tmp_path / "frame_0000.png").exists()  # no se limpia si falla


def test_api_rejects_bad_uploads() -> None:
    with TestClient(app) as client:
        r = client.post("/api/v1/render-gpx", files={"file": ("a.txt", b"hola")})
        assert r.status_code == 415
        r = client.post("/api/v1/render-gpx", files={"file": ("a.gpx", b"<gpx><trk>")})
        assert r.status_code == 422 and r.json()["error"] == "invalid_gpx"
        r = client.post("/api/v1/render-gpx", files={"file": ("a.gpx", make_gpx())}, data={"resolution": "4k"})
        assert r.status_code == 422
        r = client.post("/api/v1/render-gpx", files={"file": ("a.gpx", make_gpx())}, data={"animation_speed": "99"})
        assert r.status_code == 422


def test_api_end_to_end() -> None:
    with TestClient(app) as client:  # TestClient ejecuta las background tasks antes de devolver
        r = client.post(
            "/api/v1/render-gpx",
            files={"file": ("ruta.gpx", make_gpx(), "application/gpx+xml")},
            data={"animation_speed": "2.0", "resolution": "720p", "camera_style": "orbit"},
        )
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]

        info = client.get(f"/api/v1/jobs/{job_id}").json()
        assert info["status"] == "completed", info
        assert info["progress"] == 1.0

        points = client.get(f"/api/v1/jobs/{job_id}/track").json()["points"]
        assert len(points) == info["frame_count"]

        video = client.get(f"/api/v1/jobs/{job_id}/video")
        assert video.status_code == 200 and video.headers["content-type"] == "video/mp4"

    path = get_settings().output_dir / f"{job_id}.mp4"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    assert probe == f"h264,1280,720,yuv420p,30/1,{info['frame_count']}"
    # Los temporales del trabajo han desaparecido
    import glob, tempfile
    assert not glob.glob(os.path.join(tempfile.gettempdir(), f"gpxrender_{job_id}_*"))

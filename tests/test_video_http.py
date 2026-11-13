"""Standalone HTTP contract tests, without GPU or model weights."""
from __future__ import annotations

import base64
import io
import time
from threading import Event

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from thenoise.video.http import install_video_routes
from thenoise.video.jobs import VideoJobs
from thenoise.models.wan_video import WanVideo


def until(predicate, timeout=2):
    stop = time.monotonic() + timeout
    while time.monotonic() < stop:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("video job did not settle")


def test_video_http_t2v_and_i2v_and_content(tmp_path):
    seen = []

    def fake_render(spec, token, progress, path):
        seen.append(spec)
        progress("sampling", 1, spec.steps)
        path.write_bytes(b"\x1aE\xdf\xa3" + b"video")

    jobs = VideoJobs(fake_render, tmp_path, model="Wan2.2-TI2V-5B",
                     resolve=WanVideo.resolve_video)
    app = FastAPI()
    install_video_routes(app, jobs, "Wan2.2-TI2V-5B")
    try:
        with TestClient(app) as c:
            req = {"model": "Wan2.2-TI2V-5B", "prompt": "fox", "frames": 9,
                   "width": 480, "height": 320, "fps": 8, "duration": 10}
            r = c.post("/videos", json=req)
            assert r.status_code == 202
            ident = r.json()["id"]
            until(lambda: c.get(f"/videos/{ident}").json()["status"] == "completed")
            status = c.get(f"/videos/{ident}").json()
            assert status["frame_count"] == 9 and status["duration"] == 9/8
            assert c.get(f"/videos/{ident}/content").content.startswith(b"\x1aE\xdf\xa3")

            im = Image.new("RGB", (40, 30), color="red")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            req["image"] = base64.b64encode(buf.getvalue()).decode("ascii")
            req["output_format"] = "mp4"
            r = c.post("/videos", json=req)
            assert r.status_code == 202
            second = r.json()["id"]
            until(lambda: c.get(f"/videos/{second}").json()["status"] == "completed")
            assert seen[0].mode == "t2v" and seen[1].mode == "i2v"
            assert seen[1].image == buf.getvalue()
            assert c.get(f"/videos/{second}").json()["mime_type"] == "video/mp4"
            assert c.get("/videos/unknown").status_code == 404

            sync = c.post("/video", json={"prompt": "fox", "frames": 9,
                                          "width": 480, "height": 320,
                                          "output_format": "mp4",
                                          "lora_specs": ["style:0.7"]})
            assert sync.status_code == 200
            assert sync.headers["content-type"].startswith("video/mp4")
            assert sync.content.startswith(bytes([0x1a, 69, 0xdf, 0xa3]))
            assert seen[-1].lora_specs == ["style:0.7"]
            assert not list(tmp_path.glob("sync_*"))
            assert c.post("/video", json={"model": "wrong", "prompt": "fox"}).status_code == 400
    finally:
        jobs.close()


def test_video_http_cancel_running_and_rejection(tmp_path):
    entered, proceed = Event(), Event()

    def fake_render(spec, token, progress, path):
        entered.set()
        proceed.wait(2)
        token.check()
        path.write_bytes(b"clip")

    jobs = VideoJobs(fake_render, tmp_path, model="Wan2.2-TI2V-5B",
                     resolve=WanVideo.resolve_video)
    app = FastAPI()
    install_video_routes(app, jobs, "Wan2.2-TI2V-5B")
    try:
        with TestClient(app) as c:
            assert c.post("/videos", json={"model": "wrong", "prompt": "fox"}).status_code == 400
            assert c.post("/videos", json={"prompt": "fox", "sampler": "dpmpp"}).status_code == 400
            assert c.post("/videos", json={"prompt": "fox", "width": 7680, "height": 4320,
                                               "fps": 120, "steps": 250}).status_code == 202
            assert c.post("/videos", json={"prompt": "fox", "flow_shift": 7}).status_code == 422
            assert c.post("/videos", json={"prompt": "fox", "image": "invalid!?"}).status_code == 400
            assert c.post("/videos", json={"prompt": "fox", "image": base64.b64encode(b"not an image").decode()}).status_code == 400
            r = c.post("/videos", json={"prompt": "fox"})
            ident = r.json()["id"]
            assert entered.wait(2)
            assert c.get(f"/videos/{ident}/content").status_code == 409
            assert c.post(f"/videos/{ident}/cancel").json()["status"] == "canceling"
            proceed.set()
            until(lambda: c.get(f"/videos/{ident}").json()["status"] == "canceled")
            assert c.get(f"/videos/{ident}/content").status_code == 409
            assert c.post(f"/videos/{ident}/cancel").status_code == 409
    finally:
        jobs.close()

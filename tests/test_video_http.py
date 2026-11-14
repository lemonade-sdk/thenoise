"""Standalone HTTP contract tests, without GPU or model weights."""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from threading import Event

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from thenoise.video.http import install_video_routes
from thenoise.video.jobs import VideoJobs
from conftest import wan_adapter


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
                     resolve=wan_adapter().resolve_video)
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
            assert seen[-1].lora_specs == {"dit": ["style:0.7"]}
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
                     resolve=wan_adapter().resolve_video)
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


class _StubJobs:
    """The five routes' whole dependency, failing on command."""

    def __init__(self, failure=None):
        self.failure = failure or ValueError("nothing here")

    def submit(self, spec):
        raise self.failure

    def render_sync(self, spec):
        raise self.failure

    def status(self, video_id):
        return None

    def content(self, video_id):
        return "not_found", Path("/dev/null")

    def cancel(self, video_id):
        return False


def _client(jobs):
    app = FastAPI()
    install_video_routes(app, jobs, "Wan2.2-TI2V-5B")
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(("failure", "status", "reason"), [
    (ValueError("frames must be a 4n+1 count"), 400, "frames must be a 4n+1 count"),
    (KeyError("unexpected"), 500, "unexpected"),
])
def test_the_sync_video_route_refuses_in_text(failure, status, reason, caplog):
    """/video returns bytes, so it refuses the way the image endpoints do."""
    with _client(_StubJobs(failure)) as c:
        r = c.post("/video", json={"prompt": "fox"})
    assert r.status_code == status
    assert r.headers["content-type"].startswith("text/plain")
    assert reason in r.text
    assert ("video generation failed" in caplog.text) == (status == 500)


@pytest.mark.parametrize(("failure", "status", "reason"), [
    (ValueError("frames must be a 4n+1 count"), 400, "frames must be a 4n+1 count"),
    (OverflowError("video queue is full"), 429, "video queue is full"),
    (RuntimeError("no memory for another video"), 503, "no memory for another video"),
    (KeyError("unexpected"), 500, "unexpected"),
])
def test_the_job_api_refuses_in_json(failure, status, reason, caplog):
    """The job API answers JSON, its refusals included, and logs only what the caller
    is not to blame for.
    """
    with _client(_StubJobs(failure)) as c:
        r = c.post("/videos", json={"prompt": "fox"})
    assert r.status_code == status
    assert list(r.json()) == ["error"]
    assert reason in r.json()["error"]
    assert ("video job failed" in caplog.text) == (status == 500)


def test_a_video_job_answer_stays_json():
    """A snapshot is a JSON document with its own fields, handed out as it is."""
    class Submitted(_StubJobs):
        def submit(self, spec):
            return {"id": "v1", "status": "queued", "model": "Wan2.2-TI2V-5B",
                    "created_at": 0.0, "extra": "not handed out"}

    with _client(Submitted()) as c:
        r = c.post("/videos", json={"prompt": "fox"})
    assert r.status_code == 202
    assert r.json() == {"id": "v1", "status": "queued", "model": "Wan2.2-TI2V-5B",
                        "created_at": 0.0}


def test_reads_of_an_absent_video_answer_json():
    with _client(_StubJobs()) as c:
        for method, path, status in [("get", "/videos/nope", 404),
                                     ("get", "/videos/nope/content", 404),
                                     ("post", "/videos/nope/cancel", 409)]:
            r = getattr(c, method)(path)
            assert r.status_code == status
            assert r.json()["error"]


# NaN and infinity are outside JSON itself, so they cannot be sent to ask.
@pytest.mark.parametrize(("field", "value"), [("frames", 0), ("width", -1),
                                             ("guidance_scale", -0.5), ("duration", 0),
                                             ("lora_specs", 3), ("unknown_option", 1)])
def test_a_request_the_contract_refuses_is_a_schema_error(field, value):
    """Bounds, types and unknown fields are answered by the schema on both video routes;
    the route's own envelope carries only what the model refused.
    """
    with _client(_StubJobs()) as c:
        assert c.post("/videos", json={"prompt": "fox", field: value}).status_code == 422
        assert c.post("/video", json={"prompt": "fox", field: value}).status_code == 422

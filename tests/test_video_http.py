"""Standalone video HTTP contract tests, without GPU or model weights.

The queue underneath is the shared one (`test_jobs`); what is exercised here is the
video submit, the clip served directly, and how a resolved Wan request reads back.
"""
from __future__ import annotations

import base64
import io
import time
from threading import Event

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from thenoise.jobs import JobQueue
from thenoise.jobs_http import install_job_routes
from thenoise.video.http import VideoRequest, install_video_routes
from thenoise.video.service import video_planner
from conftest import video_pipeline

MODEL = "Wan2.2-TI2V-5B"


def until(predicate, timeout=2):
    stop = time.monotonic() + timeout
    while time.monotonic() < stop:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("video job did not settle")


class _Runtime:
    """What a video job asks of the runtime: defaults at resolve, a controller per run."""

    model_name = MODEL

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, action):
        return action(self.pipeline)


def _client(pipeline, queue):
    app = FastAPI()
    install_job_routes(app, queue)
    install_video_routes(app, queue, video_planner(_Runtime(pipeline)), MODEL)
    return TestClient(app)


def _controller(render, variant: str = "ti2v_5b"):
    class Controller:
        # The real resolve, so what the pipeline refuses is what the route answers 400.
        resolve = property(lambda self: video_pipeline(variant).resolve)

        def render(self, spec, token, progress, output):
            render(spec, token, progress, output)

    return Controller()


def _failing(failure):
    class Controller:
        def resolve(self, spec):
            raise failure

        def render(self, spec, token, progress, output):
            raise failure

    return Controller()


def test_video_t2v_and_i2v_and_content(tmp_path):
    seen = []

    def fake_render(spec, token, progress, path):
        seen.append(spec)
        progress("sampling", 1, spec.steps)
        path.write_bytes(b"\x1aE\xdf\xa3" + b"video")

    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(fake_render), queue) as c:
            req = {"model": MODEL, "prompt": "fox", "frames": 9,
                   "width": 480, "height": 320, "fps": 8, "duration": 10}
            r = c.post("/jobs/video", json=req)
            assert r.status_code == 202
            assert r.json()["kind"] == "video"
            ident = r.json()["id"]
            until(lambda: c.get(f"/jobs/{ident}").json()["status"] == "completed")
            status = c.get(f"/jobs/{ident}").json()
            assert status["resolved"]["frame_count"] == 9
            assert status["resolved"]["duration"] == 9/8
            assert c.get(f"/jobs/{ident}/content").content.startswith(b"\x1aE\xdf\xa3")

            im = Image.new("RGB", (40, 30), color="red")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            req["image"] = base64.b64encode(buf.getvalue()).decode("ascii")
            req["output_format"] = "mp4"
            r = c.post("/jobs/video", json=req)
            assert r.status_code == 202
            second = r.json()["id"]
            until(lambda: c.get(f"/jobs/{second}").json()["status"] == "completed")
            assert seen[0].mode == "t2v" and seen[1].mode == "i2v"
            assert seen[1].image == buf.getvalue()
            assert c.get(f"/jobs/{second}").json()["mime_type"] == "video/mp4"
            assert c.get("/jobs/unknown").status_code == 404

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
        queue.close()


def _b64_png(color) -> str:
    """One flat colour, as the wire carries it: base64 of PNG bytes."""
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_a_clip_that_ends_on_a_frame_of_its_own(tmp_path):
    """The I2V pair takes a last frame and answers `flf2v`."""
    seen = []

    def fake_render(spec, token, progress, path):
        seen.append(spec)
        path.write_bytes(b"\x1aE\xdf\xa3" + b"video")

    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(fake_render, "i2v_14b"), queue) as c:
            first, last = _b64_png("red"), _b64_png("blue")
            r = c.post("/jobs/video", json={"prompt": "a fox turns into a cat",
                                            "width": 480, "height": 320, "frames": 9,
                                            "image": first, "end_image": last})
            assert r.status_code == 202
            ident = r.json()["id"]
            until(lambda: c.get(f"/jobs/{ident}").json()["status"] == "completed")
            assert seen[-1].mode == "flf2v"
            assert seen[-1].image == base64.b64decode(first)
            assert seen[-1].end_image == base64.b64decode(last)
            assert c.get(f"/jobs/{ident}").json()["resolved"]["mode"] == "flf2v"

            # Refused at submit: a last frame alone, and one that is not an image.
            alone = c.post("/jobs/video", json={"prompt": "fox", "end_image": last})
            assert alone.status_code == 400 and "end_image needs image" in alone.json()["error"]
            broken = c.post("/jobs/video", json={"prompt": "fox", "image": first,
                                                 "end_image": "not base64!?"})
            assert broken.status_code == 400 and "end_image" in broken.json()["error"]
    finally:
        queue.close()


def test_references_carry_their_frame_on_the_wire():
    spec = VideoRequest(prompt="a fox, like this",
                        references=[_b64_png("red"),
                                    {"image": _b64_png("blue"), "frame": -1}]).to_spec()
    assert [guide.frame for guide in spec.references] == [None, -1]
    assert all(guide.pixels[:8] == b"\x89PNG\r\n\x1a\n" for guide in spec.references)


def test_references_are_refused_by_a_model_without_the_answer(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(lambda *a: None), queue) as c:
            r = c.post("/jobs/video", json={"prompt": "fox",
                                            "references": [_b64_png("red")]})
            assert r.status_code == 400
            assert "does not support ref2v" in r.json()["error"]
    finally:
        queue.close()


def test_a_variant_without_a_reference_mask_refuses_a_last_frame(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(lambda *a: None), queue) as c:
            r = c.post("/jobs/video", json={"prompt": "fox", "width": 480, "height": 320,
                                            "frames": 9, "image": _b64_png("red"),
                                            "end_image": _b64_png("blue")})
            assert r.status_code == 400
            assert "does not support flf2v" in r.json()["error"]
    finally:
        queue.close()


def test_video_cancel_running_and_rejection(tmp_path):
    entered, proceed = Event(), Event()

    def fake_render(spec, token, progress, path):
        entered.set()
        proceed.wait(2)
        token.check()
        path.write_bytes(b"clip")

    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(fake_render), queue) as c:
            assert c.post("/jobs/video", json={"model": "wrong", "prompt": "fox"}).status_code == 400
            assert c.post("/jobs/video", json={"prompt": "fox", "sampler": "dpmpp"}).status_code == 400
            assert c.post("/jobs/video", json={"prompt": "fox", "width": 7680, "height": 4320,
                                               "fps": 120, "steps": 250}).status_code == 202
            assert c.post("/jobs/video", json={"prompt": "fox", "flow_shift": 0}).status_code == 422
            assert c.post("/jobs/video", json={"prompt": "fox", "image": "invalid!?"}).status_code == 400
            assert c.post("/jobs/video", json={"prompt": "fox",
                                               "image": base64.b64encode(b"not an image").decode()}).status_code == 400
            r = c.post("/jobs/video", json={"prompt": "fox"})
            ident = r.json()["id"]
            assert entered.wait(2)
            assert c.get(f"/jobs/{ident}/content").status_code == 409
            assert c.post(f"/jobs/{ident}/cancel").json()["status"] == "canceling"
            proceed.set()
            until(lambda: c.get(f"/jobs/{ident}").json()["status"] == "canceled")
            assert c.get(f"/jobs/{ident}/content").status_code == 409
            assert c.post(f"/jobs/{ident}/cancel").status_code == 409
    finally:
        queue.close()


@pytest.mark.parametrize(("failure", "status", "reason"), [
    (ValueError("frames must be a 4n+1 count"), 400, "frames must be a 4n+1 count"),
    (KeyError("unexpected"), 500, "unexpected"),
])
def test_the_sync_video_route_refuses_in_text(tmp_path, failure, status, reason, caplog):
    """/video returns bytes, so it refuses the way the image endpoints do."""
    queue = JobQueue(tmp_path)
    try:
        with _client(_failing(failure), queue) as c:
            r = c.post("/video", json={"prompt": "fox"})
    finally:
        queue.close()
    assert r.status_code == status
    assert r.headers["content-type"].startswith("text/plain")
    assert reason in r.text
    assert ("video generation failed" in caplog.text) == (status == 500)


def test_a_video_job_answer_stays_json(tmp_path):
    """A snapshot is a JSON document with its own fields, handed out as it is."""
    class Controller:
        resolve = property(lambda self: video_pipeline().resolve)

        def render(self, spec, token, progress, output):
            output.write_bytes(b"clip")

    queue = JobQueue(tmp_path)
    try:
        with _client(Controller(), queue) as c:
            r = c.post("/jobs/video", json={"prompt": "fox"})
        assert r.status_code == 202
        assert list(r.json()) == ["id", "kind", "status", "model", "created_at"]
    finally:
        queue.close()


def test_a_queue_that_cannot_take_the_job_says_so_in_json(tmp_path):
    cases = [(ValueError("frames must be a 4n+1 count"), 400),
             (OverflowError("job queue is full"), 429),
             (RuntimeError("no video model is loaded"), 503),
             (KeyError("unexpected"), 500)]
    for failure, status in cases:
        queue = JobQueue(tmp_path)
        try:
            with _client(_failing(failure), queue) as c:
                r = c.post("/jobs/video", json={"prompt": "fox"})
            assert r.status_code == status
            assert list(r.json()) == ["error"]
            assert str(failure.args[0]) in r.json()["error"]
        finally:
            queue.close()

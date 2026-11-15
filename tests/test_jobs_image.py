"""The image half of the job API: what a queued generation or edit promises.

Driven end to end over the weight-free ``StubModel``, so the resolve-at-submit, the
progress a running job reports and the PNG the content route hands out are the real ones.
"""
from __future__ import annotations

import base64
import io
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image

from conftest import EditingStubModel, StubModel
from thenoise.api import create_app
from thenoise.jobs import JobQueue
from thenoise.pipeline import PipelineController
from thenoise.runtime import Runtime, Settings
from thenoise.upscale.pixel import PixelUpscalerManager


def until(predicate, timeout=5):
    stop = time.monotonic() + timeout
    while time.monotonic() < stop:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("job did not settle")


class SlowModel(StubModel):
    """Stops inside the solver, so a job can be watched and interrupted mid-sample."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.entered = threading.Event()
        self.release = threading.Event()

    def denoise_step(self, latents, t, cond, guidance_scale, i):
        self.entered.set()
        assert self.release.wait(5)
        return super().denoise_step(latents, t, cond, guidance_scale, i)


def _runtime(model=None):
    runtime = Runtime(Settings(device="cpu"))
    model = model or StubModel()
    upscalers = PixelUpscalerManager(upscaler_dir="", device="cpu")
    runtime._pipeline = PipelineController(model, upscalers)
    runtime._model = model
    runtime._model_name = model.name
    return runtime


def _client(runtime, jobs):
    return TestClient(create_app(runtime, jobs=jobs))


def _png_b64(size=(64, 64)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_a_queued_generation_is_rendered_and_handed_out(tmp_path):
    runtime, jobs = _runtime(), JobQueue(tmp_path / "jobs")
    try:
        with _client(runtime, jobs) as c:
            r = c.post("/jobs/text2image", json={"prompt": "a fox", "seed": 7})
            assert r.status_code == 202
            assert list(r.json()) == ["id", "kind", "status", "model", "created_at"]
            assert r.json()["kind"] == "image" and r.json()["model"] == "stub"
            ident = r.json()["id"]
            until(lambda: c.get(f"/jobs/{ident}").json()["status"] == "completed")

            status = c.get(f"/jobs/{ident}").json()
            resolved = status["resolved"]
            assert (resolved["width"], resolved["height"]) == (64, 64)
            assert (resolved["sampler"], resolved["mode"], resolved["seed"]) \
                == ("euler", "t2i", 7)
            assert status["mime_type"] == "image/png"
            # A job's bytes are the only image this server writes: they wait on disk for
            # the request that fetches them.
            assert [p.name for p in (tmp_path / "jobs").glob("job_*.png")] == [ident + ".png"]

            content = c.get(f"/jobs/{ident}/content")
            assert content.headers["content-type"] == "image/png"
            assert content.content[:8] == b"\x89PNG\r\n\x1a\n"
            assert Image.open(io.BytesIO(content.content)).size == (64, 64)
            # Handed out once: the file left with the response.
            assert not list((tmp_path / "jobs").glob("*.png"))
            assert c.get(f"/jobs/{ident}/content").status_code == 404
    finally:
        jobs.close()


def test_a_queued_edit_says_so_and_carries_its_reference_image(tmp_path):
    runtime, jobs = _runtime(EditingStubModel()), JobQueue(tmp_path / "jobs")
    try:
        with _client(runtime, jobs) as c:
            ident = c.post("/jobs/edit", json={"prompt": "a fox in a scarf",
                                               "image": _png_b64(), "seed": 2}).json()["id"]
            until(lambda: c.get(f"/jobs/{ident}").json()["status"] == "completed")
            status = c.get(f"/jobs/{ident}").json()
            assert status["resolved"]["mode"] == "edit"
            assert status["resolved"]["ref_method"] == "index"
            image = Image.open(io.BytesIO(c.get(f"/jobs/{ident}/content").content))
            assert image.size == (status["resolved"]["width"], status["resolved"]["height"])
    finally:
        jobs.close()


def test_a_job_reports_the_steps_it_is_taking_and_can_be_stopped(tmp_path):
    model = SlowModel()
    runtime, jobs = _runtime(model), JobQueue(tmp_path / "jobs")
    try:
        with _client(runtime, jobs) as c:
            ident = c.post("/jobs/text2image",
                           json={"prompt": "a fox", "steps": 4, "seed": 3}).json()["id"]
            assert model.entered.wait(5)
            until(lambda: c.get(f"/jobs/{ident}").json()["progress"]["stage"] == "sampling")

            assert c.post(f"/jobs/{ident}/cancel").json() == {"status": "canceling"}
            model.release.set()
            until(lambda: c.get(f"/jobs/{ident}").json()["status"] == "canceled")
            assert c.get(f"/jobs/{ident}/content").status_code == 409
            assert not list((tmp_path / "jobs").glob("*.png"))
    finally:
        jobs.close()


def test_a_request_the_model_cannot_serve_is_refused_before_it_is_queued(tmp_path):
    runtime, jobs = _runtime(), JobQueue(tmp_path / "jobs")
    try:
        with _client(runtime, jobs) as c:
            r = c.post("/jobs/edit", json={"prompt": "a fox", "image": _png_b64()})
            assert r.status_code == 400
            assert "does not support image editing" in r.json()["error"]
        assert jobs._jobs == {}
    finally:
        jobs.close()


def test_out_json_has_no_meaning_for_a_job(tmp_path):
    runtime, jobs = _runtime(), JobQueue(tmp_path / "jobs")
    try:
        with _client(runtime, jobs) as c:
            r = c.post("/jobs/text2image", json={"prompt": "a fox", "out": "json"})
            assert r.status_code == 400
            assert "out" in r.json()["error"]
        assert jobs._jobs == {}
    finally:
        jobs.close()


def test_with_nothing_loaded_a_job_is_refused_like_a_generation_is(tmp_path):
    jobs = JobQueue(tmp_path / "jobs")
    try:
        with _client(Runtime(Settings(device="cpu")), jobs) as c:
            assert c.post("/jobs/text2image", json={"prompt": "a fox"}).status_code == 503
            assert c.get("/jobs/nope").status_code == 404
            assert c.get("/jobs/nope/content").status_code == 404
            assert c.post("/jobs/nope/cancel").status_code == 409
    finally:
        jobs.close()

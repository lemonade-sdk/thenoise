import base64
import io

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from test_mesh_controller import DEFAULTS, FakeShapeModel

from thenoise.jobs import JobQueue
from thenoise.mesh.controller import MeshPipelineController
from thenoise.mesh.http import install_mesh_routes
from thenoise.mesh.service import mesh_planner

MODEL = "hunyuan3d21"


def _b64(color="red") -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color=color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Runtime:
    """What a mesh job asks of the runtime: defaults at resolve, a controller per run."""

    model_name = MODEL
    medium = "mesh"

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, action):
        return action(self.pipeline)


def _client(pipeline, queue):
    app = FastAPI()
    from thenoise.jobs_http import install_job_routes

    install_job_routes(app, queue)
    install_mesh_routes(app, queue, mesh_planner(_Runtime(pipeline)), MODEL)
    return TestClient(app)


def _controller(body=b"glTF fake"):
    real = MeshPipelineController(FakeShapeModel())

    class Controller:
        # The real resolve, so what the controller refuses is what the route answers 400.
        resolve = real.resolve

        def render(self, spec, token, progress, output):
            progress("sampling", spec.steps, spec.steps)
            output.write_bytes(body)

    return Controller()


def _failing(failure):
    class Controller:
        def resolve(self, spec):
            raise failure

        def render(self, spec, token, progress, output):
            raise failure

    return Controller()


def until(predicate, timeout=2):
    import time
    stop = time.monotonic() + timeout
    while time.monotonic() < stop:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("mesh job did not settle")


def test_a_queued_shape_comes_back_as_a_file(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/jobs/mesh", json={"model": MODEL, "image": _b64()})
            assert answer.status_code == 202
            job = answer.json()
            assert job["kind"] == "mesh"
            ident = job["id"]
            until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
            status = client.get(f"/jobs/{ident}").json()
            assert status["resolved"]["mode"] == "i2d"
            assert status["resolved"]["latents"] == DEFAULTS["num_latents"]
            assert status["resolved"]["octree_resolution"] == DEFAULTS["octree_resolution"]
            assert "image" not in status["resolved"]
            content = client.get(f"/jobs/{ident}/content")
            assert content.content == b"glTF fake"
            assert content.headers["content-type"].startswith("model/gltf-binary")
    finally:
        queue.close()


def test_the_object_container_is_offered_too(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(b"v 0 0 0"), queue) as client:
            answer = client.post("/jobs/mesh",
                                 json={"model": MODEL, "image": _b64(),
                                       "output_format": "obj", "latents": 128,
                                       "octree_resolution": 32, "seed": 5})
            ident = answer.json()["id"]
            until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
            status = client.get(f"/jobs/{ident}").json()
            assert status["resolved"]["latents"] == 128
            assert status["resolved"]["seed"] == 5
            assert status["resolved"]["output_format"] == "obj"
            assert client.get(f"/jobs/{ident}/content").content == b"v 0 0 0"
    finally:
        queue.close()


def test_the_shape_is_returned_directly_when_nobody_wants_a_job(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={"image": _b64(), "steps": 2})
            assert answer.status_code == 200
            assert answer.content == b"glTF fake"
            assert not list(tmp_path.glob("sync_*"))
    finally:
        queue.close()


def test_a_model_other_than_the_loaded_one_is_refused(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={"model": "other", "image": _b64()})
            assert answer.status_code == 400
            assert "not loaded" in answer.text
    finally:
        queue.close()


def test_a_picture_that_is_not_a_picture_is_refused_before_the_minutes_are_spent(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={"image": "not base64 at all!!"})
            assert answer.status_code == 400
            assert "base64" in answer.text or "valid image" in answer.text

            text = base64.b64encode(b"hello, not a png").decode("ascii")
            answer = client.post("/mesh", json={"image": text})
            assert answer.status_code == 400
            assert "valid image" in answer.text
    finally:
        queue.close()


def test_a_request_without_a_picture_cannot_ask_for_a_shape(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={})
            assert answer.status_code == 400
            assert "t2d" in answer.text
    finally:
        queue.close()


def test_a_controller_refusal_is_a_bad_request_not_a_crash(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_failing(ValueError("this shape is impossible")), queue) as client:
            answer = client.post("/mesh", json={"image": _b64()})
            assert answer.status_code == 400
            assert "impossible" in answer.text
            answer = client.post("/jobs/mesh", json={"image": _b64()})
            assert answer.status_code == 400
    finally:
        queue.close()


def test_a_field_nobody_offers_is_refused_rather_than_ignored(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/mesh", json={"image": _b64(), "num_frames": 4})
            assert answer.status_code == 422
    finally:
        queue.close()


def test_several_pictures_are_queued_as_the_views_of_one_shape(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/jobs/mesh",
                                 json={"model": MODEL, "image": [_b64("red"), _b64("blue")]})
            assert answer.status_code == 202
            ident = answer.json()["id"]
            until(lambda: client.get(f"/jobs/{ident}").json()["status"] == "completed")
            assert client.get(f"/jobs/{ident}").json()["resolved"]["mode"] == "mv2d"
    finally:
        queue.close()


def test_views_past_the_ones_the_model_numbers_are_refused_before_the_job(tmp_path):
    queue = JobQueue(tmp_path)
    try:
        with _client(_controller(), queue) as client:
            answer = client.post("/jobs/mesh",
                                 json={"model": MODEL, "image": [_b64()] * 5})
            assert answer.status_code == 400
            assert "at most 4 views" in answer.text
    finally:
        queue.close()

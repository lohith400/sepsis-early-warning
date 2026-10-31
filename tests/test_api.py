import pytest
from fastapi.testclient import TestClient

from app.main import app
from sepsis.samples import make_patient, to_psv


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_static_frontend(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "model" in h
    page = client.get("/")
    assert page.status_code == 200 and "Sepsis" in page.text
    assert client.get("/js/app.js").status_code == 200 and client.get("/css/styles.css").status_code == 200


def test_model_features_schema_reference(client):
    m = client.get("/api/model").json()
    for key in ("selected_features", "rules", "metrics", "curves", "history", "alert", "required_inputs"):
        assert key in m
    assert len(client.get("/api/features").json()) == 37
    assert any(f["used"] for f in client.get("/api/schema").json())
    ref = client.get("/api/reference").json()
    assert ref["models"][0]["auroc"] == 0.7547


def test_samples(client):
    ids = [s["id"] for s in client.get("/api/samples").json()]
    assert {"stable", "deteriorating", "shock"} <= set(ids)
    s = client.get("/api/samples/shock").json()
    assert len(s["hours"]) == len(s["labels"]) == 30
    assert client.get("/api/samples/nope").status_code == 404


def test_predict_json(client):
    hours, age = make_patient("deteriorating")
    r = client.post("/api/predict", json={"hours": hours, "age": age})
    assert r.status_code == 200
    body = r.json()
    assert len(body["hours"]) == 30 and body["summary"]["first_alert_hour"] is not None


def test_predict_validation(client):
    assert client.post("/api/predict", json={"hours": []}).status_code == 422
    assert client.post("/api/predict", json={"hours": [{"HR": "abc"}]}).status_code == 422
    assert client.post("/api/predict", json={"hours": [{}], "age": 500}).status_code == 422


def test_predict_psv_upload(client):
    r = client.post("/api/predict/psv", files={"file": ("p.psv", to_psv("shock"), "text/plain")})
    assert r.status_code == 200
    b = r.json()
    assert len(b["hours"]) == 30 and len(b["true_labels"]) == 30 and b["age"] == 58


def test_predict_psv_rejects_bad_files(client):
    assert client.post("/api/predict/psv", files={"file": ("x.psv", "hello world", "text/plain")}).status_code == 400
    assert client.post("/api/predict/psv", files={"file": ("x.psv", "a|b\n1|2\n", "text/plain")}).status_code == 400


def test_unknown_api_route_is_json_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404 and r.json()["detail"] == "Not found"


def test_no_model_gives_clear_503_and_never_falls_back(client, monkeypatch):
    """Without a trained model the API must refuse, not invent results."""
    monkeypatch.setattr(app.state, "model", None)
    assert client.get("/api/health").json()["status"] == "no_model"
    for url in ("/api/model", "/api/features", "/api/schema"):
        assert client.get(url).status_code == 503
    assert client.post("/api/predict", json={"hours": [{}]}).status_code == 503


def test_find_model_path_has_no_fallback(monkeypatch, tmp_path):
    from app import main
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setattr(main, "ROOT", tmp_path)          # an empty project folder
    assert main.find_model_path() is None

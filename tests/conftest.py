"""Test fixtures. Tests use a tiny SYNTHETIC model built on the fly; it is never shipped or used by the app."""
import os

import pytest

from synthetic import build_test_model


@pytest.fixture(scope="session", autouse=True)
def test_model_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("model") / "synthetic_test_model.json")
    build_test_model(path)
    os.environ["MODEL_PATH"] = path
    yield path
    os.environ.pop("MODEL_PATH", None)

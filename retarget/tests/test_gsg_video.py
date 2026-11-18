"""Tests fuer die RTMLib-Anbindung (ohne Modell/Video)."""

from retarget import gsg_video


def test_backend_standard_ist_opencv(monkeypatch):
    monkeypatch.delenv("AVATAR_RTMLIB_BACKEND", raising=False)
    assert gsg_video.rtmlib_backend() == "opencv"


def test_backend_laesst_sich_auf_onnxruntime_umstellen(monkeypatch):
    monkeypatch.setenv("AVATAR_RTMLIB_BACKEND", "ONNXRuntime")
    assert gsg_video.rtmlib_backend() == "onnxruntime"


def test_backend_faellt_bei_unbekanntem_wert_auf_opencv_zurueck(monkeypatch):
    monkeypatch.setenv("AVATAR_RTMLIB_BACKEND", "quatsch")
    assert gsg_video.rtmlib_backend() == "opencv"


def test_device_standard_ist_cpu(monkeypatch):
    monkeypatch.delenv("AVATAR_RTMLIB_DEVICE", raising=False)
    assert gsg_video.rtmlib_device() == "cpu"


def test_cache_path_heisst_rtmlib_json(tmp_path):
    assert gsg_video.cache_path(tmp_path, "61").name == "61.rtmlib.json"

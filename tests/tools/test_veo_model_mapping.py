"""Veo model mapping, Google-backend pricing and the Gemini API retirement gate."""

from datetime import date
from types import SimpleNamespace

import pytest

import tools.video.veo_video as veo_module
from tools.video.veo_video import VeoVideo, _resolve_google_model


@pytest.mark.parametrize(
    ("variant", "gemini_model"),
    [
        ("veo3", "veo-3.1-generate-preview"),
        ("veo3.1", "veo-3.1-generate-preview"),
        ("veo3/fast", "veo-3.1-fast-generate-preview"),
        ("veo3.1/fast", "veo-3.1-fast-generate-preview"),
        ("veo3.1/lite", "veo-3.1-lite-generate-preview"),
    ],
)
def test_aliases_map_to_gemini_preview_and_vertex_ga(variant, gemini_model) -> None:
    assert _resolve_google_model(variant, is_vertex=False) == gemini_model
    # Vertex only has the GA model; no invented fast/lite IDs
    assert _resolve_google_model(variant, is_vertex=True) == "veo-3.1-generate-001"


@pytest.mark.parametrize("is_vertex", [False, True])
def test_unknown_model_ids_pass_through(is_vertex) -> None:
    assert _resolve_google_model("veo-9-custom", is_vertex) == "veo-9-custom"


@pytest.mark.parametrize(
    ("variant", "resolution", "per_second"),
    [
        ("veo3.1", "720p", 0.40),
        ("veo3.1", "1080p", 0.40),
        ("veo3.1", "4k", 0.60),
        ("veo3.1/fast", "720p", 0.10),
        ("veo3.1/fast", "1080p", 0.12),
        ("veo3.1/fast", "4k", 0.30),
        ("veo3.1/lite", "720p", 0.05),
        ("veo3.1/lite", "1080p", 0.08),
        # Lite has no 4k; priced at the 1080p rate
        ("veo3.1/lite", "4k", 0.08),
    ],
)
def test_google_cost_by_variant_and_resolution(variant, resolution, per_second) -> None:
    cost = VeoVideo().estimate_cost({
        "backend": "google",
        "model_variant": variant,
        "resolution": resolution,
        "duration": "8s",
    })
    assert cost == pytest.approx(8 * per_second)


def test_google_cost_defaults_to_standard_1080p() -> None:
    assert VeoVideo().estimate_cost({"backend": "google"}) == pytest.approx(3.20)


def test_fal_cost_is_unchanged() -> None:
    tool = VeoVideo()
    assert tool.estimate_cost({"backend": "fal", "duration": "8s"}) == pytest.approx(3.20)
    assert tool.estimate_cost(
        {"backend": "fal", "model_variant": "veo3.1/fast", "duration": "8s"}
    ) == pytest.approx(1.60)


class _Models:
    def __init__(self):
        self.calls = []

    def generate_videos(self, **kwargs):
        self.calls.append(kwargs)
        asset = SimpleNamespace(uri=None, video_bytes=b"VIDEO_BYTES")
        asset.save = lambda path: open(path, "wb").write(b"VIDEO_BYTES")
        response = SimpleNamespace(generated_videos=[SimpleNamespace(video=asset)])
        return SimpleNamespace(done=True, error=None, response=response)


class _Client:
    def __init__(self, vertexai):
        self.vertexai = vertexai
        self.models = _Models()
        self.files = SimpleNamespace(download=lambda **kwargs: None)


def _run(monkeypatch, tmp_path, today, vertexai=False, variant="veo3.1"):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    import tools.google_credentials as credentials

    client = _Client(vertexai)
    monkeypatch.setattr(
        credentials,
        "get_genai_client",
        lambda http_options=None, location=None: client,
    )
    monkeypatch.setattr(veo_module, "_today", lambda: today)
    monkeypatch.setattr("tools.video._shared.probe_output", lambda path: {})

    result = VeoVideo().execute({
        "backend": "google",
        "prompt": "wind moving across grassland",
        "model_variant": variant,
        "output_path": str(tmp_path / "video.mp4"),
    })
    return result, client


@pytest.mark.parametrize("variant", ["veo3.1", "veo3.1/fast", "veo3.1/lite"])
def test_gemini_api_preview_allowed_before_shutdown(monkeypatch, tmp_path, variant) -> None:
    result, client = _run(monkeypatch, tmp_path, date(2026, 10, 21), variant=variant)

    assert "retired" not in (result.error or "")
    assert client.models.calls
    assert client.models.calls[0]["model"] == _resolve_google_model(variant, False)


@pytest.mark.parametrize("variant", ["veo3.1", "veo3.1/fast", "veo3.1/lite"])
def test_gemini_api_preview_blocked_from_shutdown(monkeypatch, tmp_path, variant) -> None:
    result, client = _run(monkeypatch, tmp_path, date(2026, 10, 22), variant=variant)

    assert result.success is False
    assert "retired on the Gemini API on 2026-10-22" in result.error
    assert "gemini_omni_video" in result.error
    assert client.models.calls == []


def test_vertex_not_blocked_after_shutdown(monkeypatch, tmp_path) -> None:
    result, client = _run(monkeypatch, tmp_path, date(2026, 11, 1), vertexai=True)

    assert "retired" not in (result.error or "")
    assert client.models.calls[0]["model"] == "veo-3.1-generate-001"


def test_custom_model_id_not_blocked_after_shutdown(monkeypatch, tmp_path) -> None:
    result, client = _run(
        monkeypatch, tmp_path, date(2026, 11, 1), variant="veo-9-custom"
    )

    assert "retired" not in (result.error or "")
    assert client.models.calls[0]["model"] == "veo-9-custom"

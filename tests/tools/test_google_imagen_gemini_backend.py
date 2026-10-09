"""Tests for the Gemini image backend in google_imagen.

Gemini image models generate through generate_content with an image_config.
Since Imagen was retired on the Gemini API (2026-08-17) this is the only
Google image path; the tool keeps the `google_imagen` name for compatibility.
"""

import sys
import types as pytypes
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class _FakeInline:
    def __init__(self, data: bytes):
        self.data = data


class _FakePart:
    def __init__(self, data: bytes):
        self.inline_data = _FakeInline(data)


class _FakeContent:
    def __init__(self, parts):
        self.parts = parts


class _FakeCandidate:
    def __init__(self, parts):
        self.content = _FakeContent(parts)


class _FakeResponse:
    def __init__(self, parts):
        self.candidates = [_FakeCandidate(parts)]


@pytest.fixture
def imagen_tool(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    calls: list[dict] = []

    class _FakeModels:
        def generate_content(self, model=None, contents=None, config=None):
            calls.append({"model": model, "contents": contents, "config": config})
            return _FakeResponse([_FakePart(b"GEMINI_IMG")])

    class _FakeClient:
        models = _FakeModels()

    import tools.google_credentials as gc

    monkeypatch.setattr(
        gc, "get_genai_client", lambda http_options=None, location=None: _FakeClient()
    )

    from tools.graphics.google_imagen import GoogleImagen

    return GoogleImagen(), calls


def test_gemini_model_routes_to_generate_content(imagen_tool, tmp_path):
    tool, calls = imagen_tool
    out = tmp_path / "img.png"

    result = tool.execute(
        {
            "prompt": "a flower",
            "model": "gemini-2.5-flash-image",
            "aspect_ratio": "16:9",
            "output_path": str(out),
        }
    )

    assert result.success
    assert result.data["model"] == "gemini-2.5-flash-image"
    assert out.read_bytes() == b"GEMINI_IMG"

    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-2.5-flash-image"
    # Aspect ratio must reach the API through image_config, not be dropped.
    assert calls[0]["config"].image_config.aspect_ratio == "16:9"


def test_image_selector_maps_model_name_to_google_model(
    imagen_tool, monkeypatch, tmp_path
):
    """The governed selector must be able to reach the Gemini backend."""
    from tools.graphics.image_selector import ImageSelector

    tool, calls = imagen_tool
    selector = ImageSelector()
    monkeypatch.setattr(selector, "_providers", lambda: [tool])

    result = selector.execute(
        {
            "prompt": "a flower",
            "preferred_provider": "google_imagen",
            "model_name": "gemini-2.5-flash-image",
            "output_path": str(tmp_path / "selected.png"),
        }
    )

    assert result.success, result.error
    assert calls[0]["model"] == "gemini-2.5-flash-image"
    assert result.data["selected_tool"] == "google_imagen"
    assert result.data["model"] == "gemini-2.5-flash-image"


def test_gemini_cost_estimate_is_per_image():
    from tools.graphics.google_imagen import GoogleImagen

    tool = GoogleImagen()
    assert tool.estimate_cost(
        {"model": "gemini-2.5-flash-image", "number_of_images": 2}
    ) == pytest.approx(0.039 * 2)


def test_text_only_response_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    class _TextPart:
        inline_data = None

    class _FakeModels:
        def generate_content(self, model=None, contents=None, config=None):
            return _FakeResponse([_TextPart()])

    class _FakeClient:
        models = _FakeModels()

    import tools.google_credentials as gc

    monkeypatch.setattr(
        gc, "get_genai_client", lambda http_options=None, location=None: _FakeClient()
    )

    from tools.graphics.google_imagen import GoogleImagen

    result = GoogleImagen().execute(
        {
            "prompt": "a flower",
            "model": "gemini-2.5-flash-image",
            "output_path": str(tmp_path / "img.png"),
        }
    )

    assert not result.success
    assert "No image data" in result.error


def test_default_model_is_nano_banana_2_1(imagen_tool, tmp_path):
    from tools.graphics.google_imagen import GoogleImagen

    tool, calls = imagen_tool
    schema_model = GoogleImagen.input_schema["properties"]["model"]
    assert schema_model["default"] == "gemini-nano-banana-2.1"
    assert not any(m.startswith("imagen-") for m in schema_model["enum"])

    result = tool.execute({"prompt": "a flower", "output_path": str(tmp_path / "d.png")})
    assert result.success, result.error
    assert calls[0]["model"] == "gemini-nano-banana-2.1"
    assert result.data["cost_status"] == "estimated"
    assert result.cost_usd == pytest.approx(0.0336)
    assert tool.estimate_cost({"prompt": "x"}) == pytest.approx(0.0336)


@pytest.mark.parametrize(
    "model,resolution,expected",
    [
        ("gemini-nano-banana-2.1", None, 0.0336),
        ("gemini-nano-banana-2.1", "1K", 0.0336),
        ("gemini-nano-banana-2.1", "2K", 0.0504),
        ("gemini-nano-banana-2.1", "4K", 0.113),
        ("gemini-3-pro-image", None, 0.134),
        ("gemini-3-pro-image", "2K", 0.134),
        ("gemini-3-pro-image", "4K", 0.24),
        ("gemini-3.1-flash-image", None, 0.067),
        ("gemini-3.1-flash-image", "2K", 0.101),
        ("gemini-3.1-flash-image", "4K", 0.151),
        ("gemini-3.1-flash-lite-image", None, 0.0336),
        ("gemini-3.1-flash-lite-image", "1K", 0.0336),
        ("gemini-2.5-flash-image", None, 0.039),
    ],
)
def test_cost_per_model_and_size(model, resolution, expected):
    from tools.graphics.google_imagen import GoogleImagen

    inputs = {"model": model, "number_of_images": 3}
    if resolution:
        inputs["resolution"] = resolution
    # Known models have a real quote — no PriceQuoteRequired.
    assert GoogleImagen().estimate_cost(inputs) == pytest.approx(expected * 3)


def test_known_model_result_carries_cost(imagen_tool, tmp_path):
    tool, _ = imagen_tool
    result = tool.execute(
        {
            "prompt": "a flower",
            "model": "gemini-3-pro-image",
            "resolution": "4K",
            "output_path": str(tmp_path / "p.png"),
        }
    )
    assert result.success, result.error
    assert result.cost_usd == pytest.approx(0.24)
    assert result.data["cost_status"] == "estimated"
    assert result.data["resolution"] == "4K"


@pytest.mark.parametrize(
    "model", ["imagen-4.0-generate-001", "imagen-4.0-fast-generate-001", "imagen-4.0-ultra-generate-001"]
)
def test_retired_imagen_model_is_a_clear_error(imagen_tool, tmp_path, model):
    from tools.graphics.google_imagen import IMAGEN_RETIRED_ERROR

    tool, calls = imagen_tool
    result = tool.execute(
        {"prompt": "a flower", "model": model, "output_path": str(tmp_path / "i.png")}
    )
    assert not result.success
    assert result.error == (
        "Imagen models were retired on the Gemini API on 2026-08-17 — use "
        "gemini-nano-banana-2.1 (default) or gemini-3-pro-image."
    )
    assert result.error == IMAGEN_RETIRED_ERROR
    assert calls == []
    with pytest.raises(ValueError, match="retired"):
        tool.estimate_cost({"model": model})


def test_lite_rejects_2k_before_calling_api(imagen_tool, tmp_path):
    tool, calls = imagen_tool
    inputs = {
        "prompt": "a flower",
        "model": "gemini-3.1-flash-lite-image",
        "resolution": "2K",
        "output_path": str(tmp_path / "l.png"),
    }
    result = tool.execute(inputs)
    assert not result.success
    assert "gemini-3.1-flash-lite-image" in result.error
    assert "2K" in result.error
    assert calls == []
    with pytest.raises(ValueError, match="2K"):
        tool.estimate_cost(inputs)


def test_gemini_2_5_rejects_resolution(imagen_tool, tmp_path):
    tool, calls = imagen_tool
    result = tool.execute(
        {
            "prompt": "a flower",
            "model": "gemini-2.5-flash-image",
            "resolution": "1K",
            "output_path": str(tmp_path / "r.png"),
        }
    )
    assert not result.success
    assert "Resolution control requires Gemini 3 image models" in result.error
    assert calls == []


def test_resolution_reaches_image_config(imagen_tool, tmp_path):
    tool, calls = imagen_tool
    result = tool.execute(
        {
            "prompt": "a flower",
            "resolution": "2K",
            "output_path": str(tmp_path / "r.png"),
        }
    )
    assert result.success, result.error
    assert calls[0]["config"].image_config.image_size == "2K"


def test_unknown_model_is_rejected(imagen_tool):
    tool, calls = imagen_tool
    result = tool.execute({"prompt": "x", "model": "gemini-9-image"})
    assert not result.success
    assert "Unsupported Google image model" in result.error
    assert calls == []


def test_deprecated_model_surfaces_note(imagen_tool, tmp_path):
    tool, _ = imagen_tool
    result = tool.execute(
        {
            "prompt": "a flower",
            "model": "gemini-3.1-flash-image",
            "output_path": str(tmp_path / "x.png"),
        }
    )
    assert result.success, result.error
    assert "gemini-nano-banana-2.1" in result.data["model_note"]

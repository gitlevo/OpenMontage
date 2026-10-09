"""Regression tests: google_imagen must return every image it requests and bills for.

Result handling was once hardcoded to the first image — images 1..n-1 never
reached disk or `artifacts`, while `images_generated` and the cost claimed n.
The user paid for n images and received one.

The tool now runs exclusively on Gemini image models via generate_content
(Imagen was retired on 2026-08-17), which returns one image per call, so the
tool makes `number_of_images` calls. These tests mock the shared genai client
the same way tests/tools/test_google_imagen_gemini_backend.py does.

Mirrors tests/tools/test_openai_image_multi_output.py, which covers the same
defect class in the OpenAI provider.
"""

import sys
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
    # Stub the genai client so execute() runs fully offline; each call returns
    # a distinct image so overwrites would be detectable.
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    calls: list[dict] = []

    class _FakeModels:
        def generate_content(self, model=None, contents=None, config=None):
            calls.append({"model": model, "contents": contents, "config": config})
            return _FakeResponse([_FakePart(f"IMAGE_{len(calls) - 1}".encode())])

    class _FakeClient:
        models = _FakeModels()

    import tools.google_credentials as gc

    monkeypatch.setattr(
        gc, "get_genai_client", lambda http_options=None, location=None: _FakeClient()
    )
    from tools.graphics.google_imagen import GoogleImagen

    tool = GoogleImagen()
    tool.calls = calls
    return tool


def test_all_requested_images_are_written(imagen_tool, tmp_path):
    out = tmp_path / "gen.png"
    result = imagen_tool.execute(
        {"prompt": "p", "number_of_images": 4, "output_path": str(out)}
    )

    assert result.success, result.error
    assert result.data["images_generated"] == 4
    assert len(result.artifacts) == 4
    assert len(imagen_tool.calls) == 4

    files = sorted(tmp_path.glob("*.png"))
    assert len(files) == 4  # every image reached disk, none overwritten
    assert [f.name for f in files] == ["gen_1.png", "gen_2.png", "gen_3.png", "gen_4.png"]
    contents = {f.read_bytes() for f in files}
    assert contents == {b"IMAGE_0", b"IMAGE_1", b"IMAGE_2", b"IMAGE_3"}


def test_artifacts_match_billed_image_count(imagen_tool, tmp_path):
    # What the user pays for must equal what they receive.
    inputs = {
        "prompt": "p",
        "number_of_images": 3,
        "output_path": str(tmp_path / "img.png"),
    }
    result = imagen_tool.execute(inputs)
    billed = imagen_tool.estimate_cost(inputs)

    assert len(result.artifacts) == 3
    # Default model gemini-nano-banana-2.1 at 1K: $0.0336 per image.
    assert billed == pytest.approx(0.0336 * 3)
    assert result.cost_usd == pytest.approx(billed)


def test_single_image_keeps_exact_output_path(imagen_tool, tmp_path):
    out = tmp_path / "single.png"
    result = imagen_tool.execute(
        {"prompt": "p", "number_of_images": 1, "output_path": str(out)}
    )

    assert result.success, result.error
    assert result.artifacts == [str(out)]
    assert out.read_bytes() == b"IMAGE_0"


def test_multi_output_paths_are_suffixed_and_unique():
    from tools.graphics.google_imagen import GoogleImagen

    paths = GoogleImagen._output_paths("/tmp/art/pic.png", 3)
    assert [p.name for p in paths] == ["pic_1.png", "pic_2.png", "pic_3.png"]
    assert len(set(paths)) == 3

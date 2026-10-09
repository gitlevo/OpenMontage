"""Regression: a provider's code-level model default must match its schema's
declared `model.default`.

Bug: runway_video and higgsfield_video hardcoded stale model defaults in
estimate_cost/estimate_runtime/execute (`gen4_turbo` / `kling_3.0`) while the
schema advertised `seedance_2.0` as the premium default. Omitting `model` then
quoted the wrong (cheap) model's cost and silently generated a different model
than the schema promised — a Decision-Communication / cost-accuracy violation.
"""

import pytest

import tools.graphics.google_imagen as google_imagen
import tools.video.gemini_omni_video as gemini_omni_video
import tools.video.higgsfield_video as higgsfield_video
import tools.video.runway_video as runway_video
from tools.audio.google_music import GoogleMusic
from tools.graphics.google_imagen import GoogleImagen
from tools.video.gemini_omni_video import GeminiOmniVideo
from tools.video.higgsfield_video import HiggsFieldVideo
from tools.video.runway_video import RunwayVideo


@pytest.mark.parametrize(
    "tool_cls, module",
    [(RunwayVideo, runway_video), (HiggsFieldVideo, higgsfield_video)],
)
def test_default_model_constant_matches_schema(tool_cls, module):
    # `execute()` reads `model` via `_DEFAULT_MODEL`; locking the constant to the
    # schema default guards the silent-model-swap path without a network call.
    schema_default = tool_cls().input_schema["properties"]["model"]["default"]
    assert module._DEFAULT_MODEL == schema_default


@pytest.mark.parametrize("tool_cls", [RunwayVideo, HiggsFieldVideo])
def test_estimate_default_model_matches_schema(tool_cls):
    tool = tool_cls()
    schema_default = tool.input_schema["properties"]["model"]["default"]

    # Cost/runtime with `model` omitted must equal the schema's declared default,
    # not some stale hardcoded fallback.
    assert tool.estimate_cost({}) == tool.estimate_cost({"model": schema_default}), (
        f"{tool.name}.estimate_cost default model diverges from schema default "
        f"{schema_default!r}"
    )
    assert tool.estimate_runtime({}) == tool.estimate_runtime({"model": schema_default}), (
        f"{tool.name}.estimate_runtime default model diverges from schema default "
        f"{schema_default!r}"
    )


# ---- Google providers (2026-10 model refresh) ----

@pytest.mark.parametrize(
    "tool_cls, default_const",
    [
        (GeminiOmniVideo, lambda: gemini_omni_video._DEFAULT_MODEL),
        (GoogleImagen, lambda: google_imagen.DEFAULT_MODEL),
    ],
)
def test_google_default_model_constant_matches_schema(tool_cls, default_const):
    schema_default = tool_cls().input_schema["properties"]["model"]["default"]
    assert default_const() == schema_default


@pytest.mark.parametrize("tool_cls", [GeminiOmniVideo, GoogleImagen, GoogleMusic])
def test_google_estimate_default_model_matches_schema(tool_cls):
    tool = tool_cls()
    schema_default = tool.input_schema["properties"]["model"]["default"]
    minimal = {"prompt": "a quiet harbor at dawn"}

    # Omitting `model` must quote without raising (no PriceQuoteRequired for the
    # default) and must equal the quote for the explicit schema default.
    default_cost = tool.estimate_cost(minimal)
    assert default_cost > 0
    assert default_cost == tool.estimate_cost({**minimal, "model": schema_default}), (
        f"{tool.name}.estimate_cost default model diverges from schema default "
        f"{schema_default!r}"
    )


def test_google_music_schema_default_is_priced():
    tool = GoogleMusic()
    schema_default = tool.input_schema["properties"]["model"]["default"]
    assert schema_default in tool.MODEL_COSTS_USD
    assert tool.estimate_cost({"prompt": "x"}) == tool.MODEL_COSTS_USD[schema_default]

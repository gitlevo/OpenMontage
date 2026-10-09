"""Google image generation via Gemini image models (Nano Banana family).

The tool keeps its historical name ``google_imagen`` for compatibility, but
Imagen 4 (all variants) was shut down on the Gemini API on 2026-08-17 and is
not offered on Vertex AI either. Every request now goes through
``generate_content`` with an ``image_config`` on a Gemini image model.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.google_credentials import has_google_credentials

DEFAULT_MODEL = "gemini-nano-banana-2.1"

IMAGEN_RETIRED_ERROR = (
    "Imagen models were retired on the Gemini API on 2026-08-17 — use "
    "gemini-nano-banana-2.1 (default) or gemini-3-pro-image."
)

# Single source of truth for supported models, output sizes and per-image
# prices (USD). Sources: ai.google.dev image-generation, pricing and
# deprecations pages, checked 2026-10-09. ``sizes`` lists the accepted
# ``image_size`` values (uppercase K); an empty tuple means the model takes no
# image_size. Without an explicit resolution the 1K price applies.
_GEMINI_IMAGE_MODELS: dict[str, dict] = {
    "gemini-nano-banana-2.1": {
        "sizes": ("1K", "2K", "4K"),
        "price_by_size": {"1K": 0.0336, "2K": 0.0504, "4K": 0.113},
        "status": "ga",
        "note": "Nano Banana 2.1, GA 2026-10-06. Default: best quality per dollar.",
    },
    "gemini-3-pro-image": {
        "sizes": ("1K", "2K", "4K"),
        "price_by_size": {"1K": 0.134, "2K": 0.134, "4K": 0.24},
        "status": "ga",
        "note": "Highest fidelity and text rendering; 1K and 2K cost the same.",
    },
    "gemini-3.1-flash-image": {
        "sizes": ("1K", "2K", "4K"),
        "price_by_size": {"1K": 0.067, "2K": 0.101, "4K": 0.151},
        "status": "deprecated",
        "note": "Deprecated 2026-10-06, replaced by gemini-nano-banana-2.1.",
    },
    "gemini-3.1-flash-lite-image": {
        "sizes": ("1K",),
        "price_by_size": {"1K": 0.0336},
        "status": "ga",
        "note": "Cheapest option; 1K output only.",
    },
    "gemini-2.5-flash-image": {
        "sizes": (),
        "price_by_size": {"1K": 0.039},
        "status": "deprecated",
        "note": "No image_size control; shuts down 2027-03-15.",
    },
}

# Aspect ratio to approximate pixel dimensions (for cost/reporting only)
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "16:9": (1344, 768),
}


def _dims_to_aspect_ratio(width: int, height: int) -> str:
    """Convert width/height to the nearest supported aspect ratio."""
    target = width / height
    best = "1:1"
    best_diff = float("inf")
    for ratio, (w, h) in ASPECT_RATIOS.items():
        diff = abs(target - w / h)
        if diff < best_diff:
            best_diff = diff
            best = ratio
    return best


class GoogleImagen(BaseTool):
    name = "google_imagen"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "google_imagen"
    stability = ToolStability.BETA
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []  # checked dynamically via env var
    install_instructions = (
        "Auth option A — API key (AI Studio): set GOOGLE_API_KEY (or GEMINI_API_KEY).\n"
        "  Get one at https://aistudio.google.com/apikey\n"
        "Auth option B — service account (Vertex AI): set GOOGLE_APPLICATION_CREDENTIALS\n"
        "  to a service-account JSON key (needs the 'google-auth' package), plus\n"
        "  GOOGLE_CLOUD_PROJECT and optionally GOOGLE_CLOUD_LOCATION (default us-central1).\n"
        "  Requires the Vertex AI API enabled and billing on the project.\n"
        "Both paths serve the Gemini image models (default gemini-nano-banana-2.1)."
    )
    agent_skills = ["provider-model-refresh"]

    capabilities = ["generate_image", "generate_illustration", "text_to_image"]
    supports = {
        "negative_prompt": False,
        "seed": False,
        "custom_size": False,
        "aspect_ratio": True,
        "image_edit": True,
    }
    best_for = [
        "high-quality photorealistic images and legible in-image text",
        "editing and style references with up to 14 reference images",
        "character/product consistency across shots via reference images",
        "1K/2K/4K output on Gemini 3 image models",
    ]
    not_good_for = [
        "negative prompt control (not supported)",
        "exact pixel dimensions (uses aspect ratios)",
        "deterministic output (no seed control)",
        "offline generation",
    ]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Image description (max 480 tokens)",
            },
            "aspect_ratio": {
                "type": "string",
                "enum": ["1:1", "3:4", "4:3", "9:16", "16:9"],
                "default": "1:1",
                "description": "Aspect ratio of generated image",
            },
            "width": {
                "type": "integer",
                "description": "Desired width in pixels — mapped to nearest aspect ratio",
            },
            "height": {
                "type": "integer",
                "description": "Desired height in pixels — mapped to nearest aspect ratio",
            },
            "model": {
                "type": "string",
                "enum": list(_GEMINI_IMAGE_MODELS),
                "default": DEFAULT_MODEL,
                "description": "Gemini image model, called through generate_content. "
                "gemini-nano-banana-2.1 (default, 1K/2K/4K), gemini-3-pro-image "
                "(highest fidelity, 1K/2K/4K), gemini-3.1-flash-lite-image "
                "(cheapest, 1K only), gemini-3.1-flash-image (deprecated), "
                "gemini-2.5-flash-image (no resolution control, shuts down "
                "2027-03-15). Imagen models are retired.",
            },
            "number_of_images": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "maximum": 4,
            },
            "output_path": {"type": "string"},
            "generation_mode": {"type": "string", "enum": ["generate", "edit"]},
            "image_path": {"type": "string"},
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 14,
            },
            "resolution": {
                "type": "string",
                "enum": ["1K", "2K", "4K"],
                "description": "Output size (image_size). Supported sizes depend "
                "on the model; defaults to the model's 1K output.",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=2, retryable_errors=["rate_limit", "timeout"]
    )
    idempotency_key_fields = ["prompt", "aspect_ratio", "model"] + [
        "generation_mode",
        "image_path",
        "image_paths",
        "resolution",
        "number_of_images",
    ]
    side_effects = [
        "writes image file to output_path",
        "calls Google Gemini API (generate_content)",
    ]
    user_visible_verification = ["Inspect generated image for relevance and quality"]

    @staticmethod
    def _output_paths(output_path: str | None, count: int) -> list[Path]:
        """Derive one output path per generated image.

        With a single image, honor the requested path as-is. With several,
        suffix each with `_1`, `_2`, … so no image overwrites another.
        """
        ext = ".png"
        if not output_path:
            return [Path(f"generated_image_{idx + 1}{ext}") for idx in range(count)]

        path = Path(output_path)
        suffix = path.suffix or ext
        if count == 1:
            return [path if path.suffix else path.with_suffix(suffix)]

        base = path.with_suffix("") if path.suffix else path
        return [base.parent / f"{base.name}_{idx + 1}{suffix}" for idx in range(count)]

    def get_status(self) -> ToolStatus:
        # API key -> AI Studio endpoint; service-account JSON -> Vertex AI.
        if has_google_credentials():
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    @staticmethod
    def _validate_model_inputs(model: str, resolution: str | None) -> str | None:
        """Return an error message for an unusable model/resolution, else None."""
        if model.startswith("imagen-"):
            return IMAGEN_RETIRED_ERROR
        spec = _GEMINI_IMAGE_MODELS.get(model)
        if spec is None:
            return f"Unsupported Google image model: {model}"
        if resolution:
            if not spec["sizes"]:
                return (
                    "Resolution control requires Gemini 3 image models; "
                    f"{model} takes no image_size"
                )
            if resolution not in spec["sizes"]:
                return (
                    f"{model} does not support resolution {resolution}; "
                    f"supported: {', '.join(spec['sizes'])}"
                )
        return None

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        model = inputs.get("model", DEFAULT_MODEL)
        resolution = inputs.get("resolution")
        error = self._validate_model_inputs(model, resolution)
        if error:
            raise ValueError(error)
        n = inputs.get("number_of_images", 1)
        price = _GEMINI_IMAGE_MODELS[model]["price_by_size"][resolution or "1K"]
        return round(price * n, 6)

    def _resolve_aspect_ratio(self, inputs: dict[str, Any]) -> str:
        """Explicit aspect_ratio > derived from width/height > default 1:1."""
        if "aspect_ratio" in inputs:
            return inputs["aspect_ratio"]
        if "width" in inputs and "height" in inputs:
            import logging

            aspect_ratio = _dims_to_aspect_ratio(inputs["width"], inputs["height"])
            logging.getLogger(__name__).info(
                "google_imagen: remapped %sx%s to nearest supported aspect ratio %s",
                inputs["width"],
                inputs["height"],
                aspect_ratio,
            )
            return aspect_ratio
        return "1:1"

    def _execute_gemini(self, inputs: dict[str, Any], model: str) -> ToolResult:
        """Generate via a Gemini image model (default gemini-nano-banana-2.1).

        These models use generate_content with an image_config, and work on
        both auth paths (API key and Vertex service account) through the
        shared genai client.
        """
        start = time.time()
        try:
            from google.genai import types
            from tools.google_credentials import get_genai_client

            client = get_genai_client()
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to initialize Google GenAI client: {e}",
            )

        if (
            inputs.get("image_url")
            or inputs.get("image_urls")
            or inputs.get("mask_url")
        ):
            return ToolResult(
                success=False,
                error="This direct Gemini adapter accepts local image_path/image_paths, not image URLs or masks",
            )
        prompt = inputs["prompt"]
        aspect_ratio = self._resolve_aspect_ratio(inputs)
        number_of_images = inputs.get("number_of_images", 1)
        image_config = {"aspect_ratio": aspect_ratio}
        if inputs.get("resolution"):
            # Already validated against _GEMINI_IMAGE_MODELS in execute().
            image_config["image_size"] = inputs["resolution"]
        config = types.GenerateContentConfig(
            image_config=types.ImageConfig(**image_config)
        )
        contents = [prompt]
        try:
            import mimetypes

            paths = list(inputs.get("image_paths") or [])
            if inputs.get("image_path"):
                paths.insert(0, inputs["image_path"])
            if len(paths) > 14:
                raise ValueError("At most 14 reference images")
            if inputs.get("generation_mode") == "edit" and not paths:
                raise ValueError("Edit requires source images")
            if inputs.get("generation_mode") == "generate" and paths:
                raise ValueError("Use edit for source images")
            contents.extend(
                types.Part.from_bytes(
                    data=Path(p).read_bytes(),
                    mime_type=mimetypes.guess_type(p)[0] or "image/png",
                )
                for p in paths
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))

        image_bytes: list[bytes] = []
        try:
            for _ in range(number_of_images):
                response = client.models.generate_content(
                    model=model,
                    contents=contents if len(contents) > 1 else prompt,
                    config=config,
                )
                for part in response.candidates[0].content.parts or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        image_bytes.append(inline.data)
                        break
        except Exception as e:
            return ToolResult(
                success=False, error=f"Gemini image generation failed: {e}"
            )

        if not image_bytes:
            return ToolResult(
                success=False,
                error=f"No image data returned by {model} (text-only response).",
            )

        output_paths = self._output_paths(inputs.get("output_path"), len(image_bytes))
        outputs: list[str] = []
        for data, out_path in zip(image_bytes, output_paths):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            outputs.append(str(out_path))

        spec = _GEMINI_IMAGE_MODELS[model]
        data = {
            "provider": "google_imagen",
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output": outputs[0],
            "outputs": outputs,
            "images_generated": len(outputs),
            "cost_status": "estimated",
        }
        if inputs.get("resolution"):
            data["resolution"] = inputs["resolution"]
        if spec["status"] == "deprecated":
            data["model_note"] = spec["note"]
        # Bill for what was actually returned, not what was requested.
        return ToolResult(
            success=True,
            data=data,
            artifacts=outputs,
            cost_usd=self.estimate_cost(
                {**inputs, "model": model, "number_of_images": len(outputs)}
            ),
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        # Gemini image models go through generate_content via the shared genai
        # client, which resolves auth (API key or Vertex service account) itself.
        model = inputs.get("model", DEFAULT_MODEL)
        error = self._validate_model_inputs(model, inputs.get("resolution"))
        if error:
            return ToolResult(success=False, error=error)
        return self._execute_gemini(inputs, model)

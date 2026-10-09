---
name: gemini-omni
description: |
  Generate and conversationally edit short videos with Google Gemini Omni Flash (`gemini-omni-1.1-flash`, GA). Use when: (1) iterating on a clip with natural-language edits instead of regenerating ("make the phone invisible, keep everything else the same"), (2) generating 3-10s clips (360p-4k) with synthesized audio, rendered on-screen text, or timecoded beats, (3) binding reference images to roles with <FIRST_FRAME>/<LAST_FRAME>/<IMAGE_REF_N> prompt tags, (4) first/last-frame interpolation and seamless loops, (5) extending a clip up to 40s, (6) editing an existing uploaded video. Accessed via the `gemini_omni_video` tool using the project's GEMINI_API_KEY/GOOGLE_API_KEY — the same key as Imagen and Google TTS.
allowed-tools: Bash, Read, Write
metadata:
  openclaw:
    requires:
      env_any:
        - GEMINI_API_KEY
        - GOOGLE_API_KEY
---

# Gemini Omni Flash (Google DeepMind)

Gemini Omni is Google DeepMind's video generation **and editing** model family, announced at I/O 2026. **Gemini Omni Flash** is GA as `gemini-omni-1.1-flash` (the `gemini_omni_video` default as of 2026-10). It generates 3-10 second 24fps clips with synthesized audio via the Gemini **Interactions API** — 720p native, plus 360p and upscaled 1080p/4k. The original `gemini-omni-flash-preview` (developer access since June 30, 2026) is legacy and **shuts down 2026-10-22**; pass `model: "gemini-omni-flash-preview"` only to reproduce older results before then. Its differentiator in the OpenMontage fleet is **stateful conversational editing**: each generation returns an `interaction_id`, and a follow-up call with `previous_interaction_id` edits that video in place — no other wrapped provider can refine a clip without regenerating it.

OpenMontage wraps it as `gemini_omni_video` (native Gemini API, no gateway). It shares `GOOGLE_API_KEY`/`GEMINI_API_KEY` with `google_imagen` and `google_tts` — one key, three capabilities. Paid tier only: ~$0.10 per second of 720p output (billed as 5,792 output tokens/sec at $17.50/1M). Google publishes only per-token pricing, so 360p/1080p/4k quotes reuse that rate and are flagged `cost_status: "estimated"`.

Other documented routes are available when the direct Google key is not the
chosen provider:

| Route | OpenMontage call | Important limitation |
|-------|------------------|----------------------|
| fal.ai | `gemini_omni_fal` | T2V, I2V, reference video, and edit endpoints; no Google interaction ID is returned |
| Runway | `runway_video`, `model: "gemini_omni_flash"` | T2V/I2V/V2V; video edits accept up to five image references |
| ComfyUI Partner Node | `comfyui_video`, `model_family: "gemini_omni_flash"` | Hosted paid node; requires network, Comfy login, and credits |

Use the direct `gemini_omni_video` route for stateful conversational editing.
Gateway routes return ordinary provider tasks and cannot preserve Google's
`previous_interaction_id` workflow. The fal edit endpoint can still be iterated
by feeding each output video URL into the next edit call.

## When to pick it (and when not)

| Use it for | Prefer another provider for |
|---|---|
| Iterative refinement — generate, review, then edit the same clip in layers | One-shot cinematic hero clips (→ Seedance 2.0, see `seedance-2-0`) |
| Editing an existing/uploaded clip (restyle, add/remove objects, change text) | Native resolution above 720p (1080p/4k are upscales) |
| On-screen rendered text and word-by-word text beats | Seed-reproducible generations (no seed support) |
| Reference-image-bound subjects/styles via prompt tags | Silent clips (audio cannot be disabled — strip it in post) |
| Timecode-scheduled multi-beat clips from one prompt | Non-English narration (English only fully supported) |
| First/last-frame interpolation and seamless loops | |
| Extending a clip in 3-10s steps (40s total) | |

Route through `video_selector` for generation operations. **Editing (`edit_video`), `first_last_frame_to_video` and `extend` are direct-tool operations** — call `gemini_omni_video` from the registry, because the multi-turn interaction state lives outside the selector's model.

## Generation prompting

Describe **scene + camera + lighting + motion + audio**. Official example:

> Continuous, unbroken handheld shot of a fluffy tabby cat sitting on a sunny windowsill, looking out into a leafy garden. The cat's tail twitches slowly, and its ears rotate slightly toward ambient noises. Sunbeams illuminate dust motes in the air.

- **Force a single shot** explicitly: "In a single continuous shot," / "No scene cuts." Otherwise the model may cut between scenes.
- **Negatives go in prose** — there is no `negative_prompt` parameter: "No dialogue," "No extra sound effects."
- **No sampler controls**: system instructions, temperature, top_p, and seeds are all unsupported. The prompt is the only lever.
- **Meta-prompt for quality**: "Consider micro-detail, expression and timing to create a very rich, detailed but entirely natural scene."

### Timecode syntax

Schedule beats with bracketed ranges or natural language — this maps directly onto OpenMontage scene-plan timings:

```
[0-3s] A person is walking [3-6s] They stop and turn around
```

> "After 3 seconds, a woman enters the scene." / "At 5s the chorus starts in the background audio."

### Audio and on-screen text

Audio is synthesized automatically; direct it in the prompt: "Include calm background music," "The audio is a low tinny radio broadcast in the background." Rendered text works and can be timed:

> One word on the screen at a time: 'did, you, know, that, Omni, can, do, awesome, text?' Each word appears for 1s.

## Reference images (`<FIRST_FRAME>` / `<IMAGE_REF_N>` tags)

Pass local images via `reference_image_paths` (they are sent in order), then bind them to roles **inside the prompt** with tags. `<IMAGE_REF_N>` indexes from 0 in the order supplied:

```
in the style of <IMAGE_REF_0> a woman <IMAGE_REF_1> is walking
```

```
[0-3s] A studio fashion sequence. Starting with woman <IMAGE_REF_0>, she is
holding <IMAGE_REF_1> [3-6s] Then we see the man <IMAGE_REF_2> holding <IMAGE_REF_3>
```

- `<FIRST_FRAME>` makes an image the opening frame: `<FIRST_FRAME> a woman is walking`.
- Use high-resolution images; describe the intended motion specifically rather than "make it move."
- Say what each image *is* (product / character / style / background reference) — the model decides usage from context.

## Conversational editing (the differentiator)

**Editing prompts are the opposite of generation prompts: short and surgical.** Overly descriptive edit prompts cause unintended changes.

1. Generate the base clip (subject + scene + motion). The tool returns `interaction_id` in its result data.
2. Pass it back as `previous_interaction_id` with `operation="edit_video"` and describe **only the delta**.
3. Append **"Keep everything else the same."** to pin unmentioned elements.
4. Refine in layers — one turn for lighting, one for camera, one for action, one for audio.

Official good/bad pairs:

| Avoid | Instead |
|---|---|
| "In the video of the man sitting on the sofa, please add a small black cat..." | "Add a cat that jumps onto his lap, he begins to pet it. Keep everything else the same." |
| "Please remove the cell phone... and fill in the background so it looks like..." | "Make the phone invisible. Keep everything else the same." |

Other working edit prompts: "Make this video anime" / "Put a fashionable hat on this person" / "Change the lighting to be more dramatic" / "Change the text on the sign to say 'Omni Flash'".

**Gotcha — `store`:** editing via `previous_interaction_id` only works if the *prior* call kept the interaction server-side (`store` defaults to true in `gemini_omni_video`). Set `store=false` only for one-shot generations you will never edit.

**Editing uploaded videos:** pass `input_video_path` instead of `previous_interaction_id`; the tool uploads it via the Files API. Unavailable in the EEA, Switzerland, and the UK (editing *generated* videos works everywhere).

## First and last frame (`first_last_frame_to_video`)

Pass `first_frame_path` and `last_frame_path` (local jpg/png). The tool sends them in that order and, unless your prompt already contains `<FIRST_FRAME>`, prefixes it with `<FIRST_FRAME> is the first frame, <LAST_FRAME> is the last frame.` (`<LAST_FRAME>` must always be paired with `<FIRST_FRAME>`). Do not combine with `reference_image_path(s)`.

**Seamless loop recipe:**

1. Use the **same still** for both `first_frame_path` and `last_frame_path`.
2. Ask for **slow, gentle camera motion** (slow push-in, subtle drift) and small ambient motion — big moves make the return to the start frame look forced.
3. Add **"No dialogue."** (and "No music." if you will score it yourself) to the prompt.
4. Audio is always generated. For a silent loop, strip it: `ffmpeg -i loop.mp4 -c:v copy -an loop_silent.mp4`.

## Resolution

`resolution`: `360p`, `720p` (default, native), `1080p` or `4k` (upscaled). Only 720p has a derivable price (~$0.10/s); other values are quoted at the same rate and marked estimated. Prefer 720p and upscale in post unless you need Google's upscaler.

## Extending a clip (`extend`)

`operation: "extend"` with `previous_interaction_id` (a stored generation) or `input_video_path` (an uploaded clip of **≤10s**). Describe what happens next. Each call appends **3-10s**; the total caps at **40s**. Extending uploaded videos is unavailable in the EEA, Switzerland, and the UK (extending generated clips works everywhere). Chain extends by feeding each result's `interaction_id` into the next call.

## Hard limitations (as of 2026-10)

- Output: 3-10s per call (extend to 40s total), 720p native (360p, upscaled 1080p/4k), 24fps, MP4 with audio that cannot be disabled; aspect ratio `16:9` or `9:16`. All output carries an invisible SynthID watermark.
- No seed, negative prompt, temperature, top_p, or system instructions.
- No voice editing.
- Audio reference inputs unsupported. Video references ≤3s were accepted but **not processed correctly** on the preview — don't rely on them.
- Multi-video prompting unsupported; may degrade output.
- English fully supported; other languages untested.
- Images of minors (EEA/CH/UK) and certain recognizable people are blocked for upload/editing.

## Sources

- Generation & editing guide: https://ai.google.dev/gemini-api/docs/omni
- Model card: https://ai.google.dev/gemini-api/docs/models/gemini-omni-flash
- Deprecations (preview shutdown 2026-10-22): https://ai.google.dev/gemini-api/docs/deprecations
- Pricing: https://ai.google.dev/gemini-api/docs/pricing
- Announcement: https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-omni/

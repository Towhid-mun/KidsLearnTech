"""AnimateDiff-backed video generation."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:
    import torch
    from diffusers import (
        AnimateDiffPipeline,
        MotionAdapter,
        EulerDiscreteScheduler,
    )
    from moviepy.editor import ImageSequenceClip
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    AnimateDiffPipeline = None  # type: ignore[assignment]
    MotionAdapter = None  # type: ignore[assignment]
    EulerDiscreteScheduler = None  # type: ignore[assignment]
    ImageSequenceClip = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class AnimateDiffGenerator:
    """Small wrapper that prepares an AnimateDiff pipeline and renders clips."""

    def __init__(self, media_dir: Path) -> None:
        self.media_dir = media_dir
        self._pipeline: AnimateDiffPipeline | None = None
        self.available = False
        self.device = None
        self.base_model = os.getenv("ANIMATEDIFF_BASE_MODEL", "SG161222/Realistic_Vision_V5.1")
        self.motion_adapter_repo = os.getenv(
            "ANIMATEDIFF_MOTION_ADAPTER", "guoyww/animatediff-motion-adapter-v1-5-2"
        )
        self.negative_prompt = os.getenv(
            "ANIMATEDIFF_NEGATIVE_PROMPT",
            "blurry, low resolution, distorted, extra limbs, watermark",
        )
        self.guidance_scale = float(os.getenv("ANIMATEDIFF_GUIDANCE", "7.0"))
        self.steps = int(os.getenv("ANIMATEDIFF_STEPS", "25"))
        self.num_frames = int(os.getenv("ANIMATEDIFF_FRAMES", "16"))
        self.fps = int(os.getenv("ANIMATEDIFF_FPS", "8"))
        self.height = int(os.getenv("ANIMATEDIFF_HEIGHT", "512"))
        self.width = int(os.getenv("ANIMATEDIFF_WIDTH", "512"))
        self._raw_suffix = os.getenv("ANIMATEDIFF_RAW_SUFFIX", "raw")

    def startup(self) -> None:
        if self.available:
            return
        if not torch or not AnimateDiffPipeline:
            logger.warning(
                "AnimateDiff dependencies missing. Install torch/diffusers to enable the backend."
            )
            return

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token = os.getenv("HUGGINGFACE_TOKEN")
        try:
            logger.info("Loading AnimateDiff motion adapter %s", self.motion_adapter_repo)
            motion_adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_repo,
                torch_dtype=dtype,
                use_auth_token=token,
            )
            logger.info("Loading base model %s", self.base_model)
            pipeline = AnimateDiffPipeline.from_pretrained(
                self.base_model,
                motion_adapter=motion_adapter,
                torch_dtype=dtype,
                use_auth_token=token,
            )
            pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
            pipeline.set_progress_bar_config(disable=True)
            if self.device == "cuda":
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(self.device)
        except Exception as exc:  # pragma: no cover - heavy dependency setup
            logger.exception("Failed to load AnimateDiff pipeline: %s", exc)
            return

        self._pipeline = pipeline
        self.available = True
        logger.info("AnimateDiff backend ready on %s", self.device)

    def shutdown(self) -> None:
        self._pipeline = None
        self.available = False

    def generate(self, sections: Sequence[str], identifier: str) -> Path:
        if not self.available or not self._pipeline:
            raise RuntimeError("AnimateDiff backend is not initialized.")
        if not ImageSequenceClip:
            raise RuntimeError("moviepy is required to compile AnimateDiff frames.")

        prompts = [prompt.strip() for prompt in sections if prompt.strip()]
        if not prompts:
            prompts = ["Children learning together in a bright classroom"]

        prompt = " cinematic, ".join(prompts)
        positive_suffix = os.getenv(
            "ANIMATEDIFF_PROMPT_SUFFIX",
            "storybook illustration, cheerful colors, volumetric lighting, 4k, ultra-detailed",
        )
        full_prompt = f"{prompt}. {positive_suffix}".strip()

        logger.info(
            "Generating AnimateDiff clip for %s (%d frames, %dfps)",
            identifier,
            self.num_frames,
            self.fps,
        )

        result = self._pipeline(
            full_prompt,
            negative_prompt=self.negative_prompt,
            num_frames=self.num_frames,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            height=self.height,
            width=self.width,
        )

        frames = self._extract_frames(result.frames)
        if not frames:
            raise RuntimeError("AnimateDiff did not return any frames.")

        arrays = [np.array(frame) for frame in frames]
        raw_path = self.media_dir / f"video_{identifier}_{self._raw_suffix}.mp4"

        clip = ImageSequenceClip(arrays, fps=self.fps)
        clip.write_videofile(
            str(raw_path),
            fps=self.fps,
            codec="libx264",
            audio=False,
        )
        clip.close()
        return raw_path

    @staticmethod
    def _extract_frames(frames_output) -> List[object]:
        if isinstance(frames_output, list):
            if frames_output and isinstance(frames_output[0], list):
                return frames_output[0]
            return frames_output
        return []

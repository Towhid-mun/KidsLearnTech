import asyncio
import json
import logging
import os
import random
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gtts import gTTS
from moviepy import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

from video_generation import VideoGenerator

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)

VIDEO_BACKEND = os.getenv("VIDEO_GENERATOR", "slides")
video_generator = VideoGenerator(media_dir=MEDIA_DIR, backend=VIDEO_BACKEND)

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. API requests will fail until it is configured.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="ChildEdu Prototype")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=MEDIA_DIR), name="media")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
async def _startup_video_backend() -> None:
    await _run_in_thread(video_generator.startup)


@app.on_event("shutdown")
async def _shutdown_video_backend() -> None:
    video_generator.shutdown()


class LessonRequest(BaseModel):
    grade: str
    topic: str


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()



def _normalize_keyword_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        candidates = re.split(r"[,/|]", value)
    elif isinstance(value, (list, tuple, set)):
        candidates = value
    else:
        return []
    keywords: List[str] = []
    for candidate in candidates:
        text = str(candidate).strip().lower()
        if text:
            keywords.append(text)
    return keywords



def _coerce_visual_hints(raw_hints) -> List[Dict[str, object]]:
    if not raw_hints:
        return []
    if isinstance(raw_hints, dict):
        nested = (
            raw_hints.get("items")
            or raw_hints.get("steps")
            or raw_hints.get("segments")
            or list(raw_hints.values())
        )
        if nested:
            return _coerce_visual_hints(nested)
    if not isinstance(raw_hints, list):
        return []

    hints: List[Dict[str, object]] = []
    for entry in raw_hints:
        if isinstance(entry, dict):
            title = str(
                entry.get("title")
                or entry.get("label")
                or entry.get("name")
                or entry.get("step")
                or ""
            ).strip()
            prompt = str(
                entry.get("visual_prompt")
                or entry.get("prompt")
                or entry.get("text")
                or entry.get("summary")
                or title
            ).strip()
            keywords = _normalize_keyword_list(
                entry.get("keywords") or entry.get("tags") or entry.get("theme")
            )
            hints.append({"title": title, "visual_prompt": prompt, "keywords": keywords})
        elif isinstance(entry, str):
            text_entry = entry.strip()
            if text_entry:
                hints.append({"title": text_entry, "visual_prompt": text_entry, "keywords": []})
    return hints


def _generate_plan_and_contexts(grade: str, topic: str) -> dict:
    if not client:
        raise RuntimeError("OpenAI client is not configured.")

    logger.info("Requesting learning plan for grade '%s' topic '%s'", grade, topic)
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You design engaging lesson plans for kids. Produce a JSON object with the keys: "
                    "plan_overview (string, 2 sentences), learning_steps (array of 3-5 short steps), "
                    "audio_script (friendly narration ~170 words), video_sections (array of 3-6 short titles), "
                    "and visual_hints (array of 3-6 objects with title, visual_prompt, keywords describing the scene). "
                    "Make sure steps and sections are age appropriate and easy to read."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Grade: {grade}\nTopic the child wants to learn: {topic}."
                ).format(grade=grade, topic=topic),
            },
        ],
        max_tokens=700,
    )

    try:
        payload = json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode lesson plan: {exc}") from exc

    plan_overview = payload.get("plan_overview") or payload.get("learning_plan")
    audio_script = payload.get("audio_script")
    learning_steps = payload.get("learning_steps") or []
    video_sections = payload.get("video_sections") or []
    raw_visual_hints = (
        payload.get("visual_hints")
        or payload.get("visualSections")
        or payload.get("visual_prompts")
        or payload.get("visualPrompts")
        or []
    )
    visual_hints = _coerce_visual_hints(raw_visual_hints)

    if not plan_overview or not audio_script:
        raise RuntimeError("Plan overview or audio script missing from AI response.")

    def _clean_list(items):
        if isinstance(items, list):
            return [str(item).strip() for item in items if str(item).strip()]
        if isinstance(items, str):
            return [part.strip() for part in items.split("\n") if part.strip()]
        return []

    learning_steps = _clean_list(learning_steps)
    video_sections = _clean_list(video_sections)

    if not video_sections:
        video_sections = ["Let's learn together!", plan_overview]

    logger.info(
        "Plan received with %d steps and %d video sections",
        len(learning_steps),
        len(video_sections),
    )

    return {
        "plan_overview": plan_overview.strip(),
        "learning_steps": learning_steps,
        "audio_script": audio_script.strip(),
        "video_sections": video_sections[:6],
        "visual_hints": visual_hints[:6],
    }

THEME_KEYWORDS = {
    "space": {"space", "planet", "moon", "star", "galaxy", "rocket"},
    "nature": {"nature", "tree", "forest", "flower", "garden", "earth", "soil", "seed", "plant"},
    "ocean": {"ocean", "sea", "water", "river", "fish", "wave"},
    "math": {"math", "number", "count", "shape", "geometry", "fraction", "add", "plus", "sum"},
    "animals": {"animal", "dog", "cat", "lion", "bear", "bird", "dinosaur", "bug"},
    "story": {"story", "book", "read", "write", "character", "adventure"},
}

THEME_BACKGROUNDS = {
    "space": (18, 24, 55),
    "nature": (186, 232, 198),
    "ocean": (138, 204, 255),
    "math": (238, 232, 255),
    "animals": (255, 236, 214),
    "story": (255, 247, 226),
    "default": (233, 244, 255),
}


def _detect_theme(text: str) -> str:
    normalized = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(word in normalized for word in keywords):
            return theme
    if re.search(r"\d", normalized):
        return "math"
    return "default"


def _draw_theme_scene(draw: ImageDraw.ImageDraw, theme: str, width: int, height: int, rng: random.Random) -> None:
    if theme == "space":
        for _ in range(30):
            x = rng.randint(20, width - 20)
            y = rng.randint(20, int(height * 0.7))
            radius = rng.randint(1, 4)
            draw.ellipse((x, y, x + radius, y + radius), fill=(255, 255, 210))
        planet_radius = 120
        center_x = width - 260
        center_y = int(height * 0.4)
        draw.ellipse((center_x - planet_radius, center_y - planet_radius, center_x + planet_radius, center_y + planet_radius), fill=(90, 130, 255), outline=(200, 210, 255), width=6)
    elif theme == "nature":
        horizon = int(height * 0.7)
        draw.rectangle((0, horizon, width, height), fill=(140, 210, 140))
        sun_radius = 60
        draw.ellipse((width - 220, 60, width - 220 + sun_radius * 2, 60 + sun_radius * 2), fill=(255, 223, 128))
    elif theme == "ocean":
        wave_top = int(height * 0.45)
        draw.rectangle((0, wave_top, width, height), fill=(20, 120, 210))
        for wave in range(5):
            y = wave_top + wave * 45
            draw.arc((40, y, width - 40, y + 90), start=0, end=180, fill=(255, 255, 255), width=2)
    elif theme == "math":
        draw.rectangle((80, 160, width - 80, height - 200), outline=(200, 210, 235), width=4)
        draw.text((140, 200), "1 + 2 = 3", fill=(80, 60, 160), font=_get_font(58))
        draw.text((140, 300), "7 - 4 = 3", fill=(30, 130, 150), font=_get_font(52))
    elif theme == "animals":
        meadow = int(height * 0.7)
        draw.rectangle((0, meadow, width, height), fill=(190, 235, 190))
    elif theme == "story":
        table = int(height * 0.7)
        draw.rectangle((0, table, width, height), fill=(205, 170, 125))
    else:
        draw.rectangle((0, height - 180, width, height), fill=(205, 180, 150))


def _render_slide_frame(text: str, highlight: bool = False) -> np.ndarray:
    width, height = 1280, 720
    cleaned = text or "Learning Time"
    theme = _detect_theme(cleaned)
    background = THEME_BACKGROUNDS.get(theme, THEME_BACKGROUNDS["default"])
    image = Image.new("RGB", (width, height), color=background)
    draw = ImageDraw.Draw(image)
    rng = random.Random(hash(cleaned) & 0xFFFFFFFF)
    _draw_theme_scene(draw, theme, width, height, rng)

    font_size = 58 if highlight else 42
    font = _get_font(font_size)
    wrap_width = 18 if highlight else 26

    paragraphs: List[str] = []
    for paragraph in cleaned.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraphs.extend(textwrap.wrap(paragraph, width=wrap_width))
        paragraphs.append("")
    if paragraphs and paragraphs[-1] == "":
        paragraphs.pop()

    y = 180 if highlight else 110
    line_spacing = 18
    for line in paragraphs:
        if not line:
            y += line_spacing * 2
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        x = (width - line_width) / 2 if highlight else 80
        draw.text((x, y), line, fill=(20, 40, 80), font=font)
        y += line_height + line_spacing

    return np.array(image)


def _generate_video_frames(sections: List[str]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    cleaned = [section.strip() for section in sections if section.strip()]
    if not cleaned:
        cleaned = ["Learning Time", "Fun facts ahead!"]
    for idx, text in enumerate(cleaned):
        frames.append(_render_slide_frame(text, highlight=(idx == 0)))
    return frames


def _synthesize_audio(text: str, identifier: str) -> Path:
    audio_path = MEDIA_DIR / f"audio_{identifier}.mp3"
    logger.info("Generating audio file %s", audio_path.name)
    tts = gTTS(text, lang="en")
    tts.save(str(audio_path))
    return audio_path

def _assemble_video(frames: List[np.ndarray], audio_path: Path, identifier: str) -> Path:
    logger.info("Composing video visuals with %d frames", len(frames))
    audio_clip = AudioFileClip(str(audio_path))
    duration = max(audio_clip.duration, 2)

    video_path = MEDIA_DIR / f"video_{identifier}.mp4"
    segments = []
    try:
        per_slide = max(duration / max(len(frames), 1), 4)
        if not frames:
            frames = [_render_slide_frame("Learning Time", highlight=True)]
        for frame in frames:
            clip = ImageClip(frame).with_duration(per_slide)
            segments.append(clip)

        video_clip = concatenate_videoclips(segments, method="compose")
        video_clip = video_clip.with_duration(duration).with_audio(audio_clip)

        logger.info("Writing merged video %s (duration %.2fs)", video_path.name, duration)
        video_clip.write_videofile(
            str(video_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
        )
    finally:
        audio_clip.close()
        for clip in segments:
            clip.close()
        try:
            video_clip.close()  # type: ignore[name-defined]
        except Exception:
            pass

    return video_path

async def _run_in_thread(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args))

@app.post("/api/generate-lesson")
async def generate_lesson(payload: LessonRequest):
    grade = payload.grade.strip()
    topic = payload.topic.strip()
    if not grade or not topic:
        raise HTTPException(status_code=400, detail="Grade and topic are required.")

    identifier = uuid4().hex
    logger.info("Starting generation pipeline %s for grade '%s' topic '%s'", identifier, grade, topic)

    try:
        plan = _generate_plan_and_contexts(grade, topic)
        audio_task = asyncio.create_task(_run_in_thread(_synthesize_audio, plan["audio_script"], identifier))
        frames_task = asyncio.create_task(_run_in_thread(_generate_video_frames, plan["video_sections"]))
        audio_path, frames = await asyncio.gather(audio_task, frames_task)
        video_path = await _run_in_thread(_assemble_video, frames, audio_path, identifier)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to generate lesson for %s", identifier)
        for path in [MEDIA_DIR / f"audio_{identifier}.mp3", MEDIA_DIR / f"video_{identifier}.mp4"]:
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Could not remove file %s", path)
        return JSONResponse(status_code=500, content={"ok": False, "error": str(exc)})

    logger.info("Successfully generated lesson %s", identifier)
    return {
        "ok": True,
        "grade": grade,
        "topic": topic,
        "plan_overview": plan["plan_overview"],
        "learning_steps": plan["learning_steps"],
        "audio_context": plan["audio_script"],
        "video_context": plan["video_sections"],
        "video_url": f"/media/{video_path.name}",
        "audio_url": f"/media/{audio_path.name}",
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"ok": False, "error": exc.detail})




import asyncio
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gtts import gTTS
from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MEDIA_DIR = BASE_DIR / "media"
MEDIA_DIR.mkdir(exist_ok=True)

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
                    "audio_script (friendly narration ~170 words), and video_sections (array of 3-6 "
                    "short titles or sentences to show on screen). Make sure steps and sections are "
                    "age appropriate and easy to read."
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
    }

def _render_slide_frame(text: str, highlight: bool = False) -> np.ndarray:
    width, height = 1280, 720
    background = (168, 210, 255) if highlight else (235, 243, 255)
    image = Image.new("RGB", (width, height), color=background)
    draw = ImageDraw.Draw(image)

    font_size = 58 if highlight else 42
    font = _get_font(font_size)
    wrap_width = 18 if highlight else 26

    paragraphs: List[str] = []
    for paragraph in text.split("\n"):
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

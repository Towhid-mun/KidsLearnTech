from pathlib import Path

path = Path(r"C:\Working\ChildEdu\main.py")
text = path.read_text()

old_import = "from uuid import uuid4\n\nimport numpy as np"
new_import = "from uuid import uuid4\n\nimport random\n\nimport numpy as np"

old_render = '''def _render_slide_frame(text: str, highlight: bool = False) -> np.ndarray:
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
'''

new_render = '''THEME_KEYWORDS = {
    "space": {"space", "planet", "moon", "star", "galaxy", "rocket", "astronaut"},
    "nature": {"nature", "tree", "forest", "flower", "garden", "earth", "soil", "seed"},
    "ocean": {"ocean", "sea", "water", "river", "fish", "wave", "pond"},
    "math": {"math", "number", "count", "shape", "geometry", "fraction", "measure"},
    "animals": {"animal", "bear", "lion", "tiger", "dog", "cat", "bird", "dinosaur", "bug"},
    "weather": {"weather", "rain", "storm", "season", "cloud", "wind", "snow", "sun"},
    "music": {"music", "song", "instrument", "melody", "beat"},
    "story": {"story", "read", "writing", "book", "character", "adventure"},
}

THEME_BACKGROUNDS = {
    "space": (15, 20, 45),
    "nature": (190, 235, 195),
    "ocean": (140, 210, 255),
    "math": (242, 232, 255),
    "animals": (255, 235, 210),
    "weather": (210, 230, 255),
    "music": (248, 232, 255),
    "story": (255, 247, 224),
    "default": (233, 244, 255),
}


def _cartoon_theme_from_text(text: str) -> str:
    normalized = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return theme
    return "default"


def _draw_space_scene(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    for _ in range(40):
        x = rng.randint(20, width - 20)
        y = rng.randint(20, int(height * 0.7))
        radius = rng.randint(1, 4)
        draw.ellipse((x, y, x + radius, y + radius), fill=(255, 255, 210))

    planet_radius = 130
    planet_center = (width - 280, int(height * 0.45))
    draw.ellipse(
        (
            planet_center[0] - planet_radius,
            planet_center[1] - planet_radius,
            planet_center[0] + planet_radius,
            planet_center[1] + planet_radius,
        ),
        fill=(90, 110, 255),
        outline=(210, 220, 255),
        width=6,
    )
    ring_height = 30
    draw.ellipse(
        (
            planet_center[0] - planet_radius - 60,
            planet_center[1] - ring_height,
            planet_center[0] + planet_radius + 60,
            planet_center[1] + ring_height,
        ),
        outline=(230, 230, 255),
        width=4,
    )

    rocket_base_x = int(width * 0.18)
    rocket_top_y = int(height * 0.35)
    body_width = 80
    body_height = 210
    draw.rectangle(
        (
            rocket_base_x,
            rocket_top_y,
            rocket_base_x + body_width,
            rocket_top_y + body_height,
        ),
        fill=(235, 235, 235),
        outline=(255, 90, 90),
        width=4,
    )
    draw.polygon(
        [
            (rocket_base_x, rocket_top_y),
            (rocket_base_x + body_width // 2, rocket_top_y - 70),
            (rocket_base_x + body_width, rocket_top_y),
        ],
        fill=(255, 120, 120),
        outline=(255, 90, 90),
    )
    window_radius = 26
    window_center = (rocket_base_x + body_width // 2, rocket_top_y + 70)
    draw.ellipse(
        (
            window_center[0] - window_radius,
            window_center[1] - window_radius,
            window_center[0] + window_radius,
            window_center[1] + window_radius,
        ),
        fill=(90, 140, 255),
        outline=(255, 255, 255),
        width=3,
    )
    flame_points = [
        (rocket_base_x + 10, rocket_top_y + body_height),
        (rocket_base_x + body_width // 2, rocket_top_y + body_height + 70),
        (rocket_base_x + body_width - 10, rocket_top_y + body_height),
    ]
    draw.polygon(flame_points, fill=(255, 160, 60), outline=(255, 130, 30))


def _draw_nature_scene(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    horizon = int(height * 0.65)
    draw.rectangle((0, horizon, width, height), fill=(120, 200, 120))
    sun_radius = 60
    draw.ellipse(
        (width - 220, 50, width - 220 + sun_radius * 2, 50 + sun_radius * 2),
        fill=(255, 223, 128),
    )
    for offset in range(3):
        base_x = 80 + offset * 220
        base_y = 130 + rng.randint(-20, 20)
        draw.ellipse((base_x, base_y, base_x + 180, base_y + 70), fill=(255, 255, 255))

    for idx in range(2):
        trunk_x = 190 + idx * 260
        trunk_top = horizon - 190
        draw.rectangle((trunk_x, trunk_top, trunk_x + 45, horizon), fill=(140, 92, 40))
        draw.ellipse(
            (trunk_x - 60, trunk_top - 90, trunk_x + 105, trunk_top + 40),
            fill=(80, 170, 90),
        )

    for flower in range(6):
        center_x = 120 + flower * 150
        center_y = horizon + 40 + (flower % 2) * 20
        petal_color = rng.choice([(255, 131, 176), (255, 207, 134), (150, 210, 255)])
        offsets = [(-24, 0), (24, 0), (0, -24), (0, 24)]
        for dx, dy in offsets:
            draw.ellipse(
                (center_x + dx - 12, center_y + dy - 12, center_x + dx + 12, center_y + dy + 12),
                fill=petal_color,
            )
        draw.ellipse((center_x - 9, center_y - 9, center_x + 9, center_y + 9), fill=(255, 255, 255))


def _draw_ocean_scene(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    wave_top = int(height * 0.45)
    draw.rectangle((0, wave_top, width, height), fill=(20, 120, 210))
    for wave in range(5):
        y = wave_top + wave * 45
        draw.arc((40, y, width - 40, y + 90), start=0, end=180, fill=(255, 255, 255), width=2)

    fish_colors = [(255, 140, 120), (255, 214, 120), (130, 210, 255)]
    for _ in range(4):
        fish_x = rng.randint(100, width - 220)
        fish_y = rng.randint(wave_top + 20, height - 150)
        body_box = (fish_x, fish_y, fish_x + 140, fish_y + 60)
        color = rng.choice(fish_colors)
        draw.ellipse(body_box, fill=color)
        tail = [
            (fish_x + 140, fish_y + 30),
            (fish_x + 170, fish_y),
            (fish_x + 170, fish_y + 60),
        ]
        draw.polygon(tail, fill=color)
        draw.ellipse((fish_x + 30, fish_y + 22, fish_x + 44, fish_y + 36), fill=(255, 255, 255))
        draw.ellipse((fish_x + 34, fish_y + 26, fish_x + 40, fish_y + 32), fill=(25, 40, 70))

    for coral in range(5):
        base_x = 80 + coral * 220
        base_y = height - 90
        draw.line((base_x, height - 40, base_x, base_y), fill=(255, 120, 160), width=10)
        draw.line((base_x, base_y, base_x - 35, base_y - 45), fill=(255, 120, 160), width=8)
        draw.line((base_x, base_y - 10, base_x + 35, base_y - 50), fill=(255, 120, 160), width=8)


def _draw_math_scene(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    for i in range(4):
        x = 160 + i * 220
        draw.line((x, 140, x, height - 220), fill=(210, 210, 230), width=2)
    for i in range(3):
        y = 140 + i * 120
        draw.line((120, y, width - 120, y), fill=(210, 210, 230), width=2)

    equations = ["1 + 2 = 3", "7 - 4 = 3", "10 ÷ 2 = 5"]
    colors = [(80, 60, 160), (20, 130, 150), (200, 120, 50)]
    for idx, eq in enumerate(equations):
        font = _get_font(64)
        x = 200 + idx * 320
        y = height - 340
        draw.text((x, y), eq, fill=colors[idx], font=font)

    draw.rectangle((220, 220, 360, 360), outline=(255, 120, 120), width=6)
    draw.ellipse((420, 220, 560, 360), outline=(90, 170, 90), width=6)
    draw.polygon([(640, 360), (720, 220), (800, 360)], outline=(80, 120, 200), width=6)


def _draw_animal_scene(draw: ImageDraw.ImageDraw, width: int, height: int, rng: random.Random) -> None:
    meadow = int(height * 0.62)
    draw.rectangle((0, meadow, width, height), fill=(176, 222, 164))
    colors = [(255, 222, 200), (255, 238, 170), (210, 220, 255)]
    for idx in range(3):
        center_x = 220 + idx * 280
        center_y = meadow - 40
        color = colors[idx % len(colors)]
        head = (center_x - 90, center_y - 90, center_x + 90, center_y + 90)
        draw.ellipse(head, fill=color, outline=(120, 90, 60), width=4)
        draw.ellipse(
            (center_x - 70, center_y - 130, center_x - 30, center_y - 90),
            fill=color,
            outline=(120, 90, 60),
            width=3,
        )
        draw.ellipse(
            (center_x + 30, center_y - 130, center_x + 70, center_y - 90),
            fill=color,
            outline=(120, 90, 60),
            width=3,
        )
        draw.ellipse((center_x - 30, center_y - 20, center_x - 10, center_y), fill=(40, 40, 60))
        draw.ellipse((center_x + 10, center_y - 20, center_x + 30, center_y), fill=(40, 40, 60))
        draw.ellipse((center_x - 12, center_y + 10, center_x + 12, center_y + 34), fill=(90, 60, 40))
        draw.arc((center_x - 60, center_y + 20, center_x + 60, center_y + 80), start=200, end=340, fill=(90, 60, 40), width=4)
        draw.ellipse((center_x - 60, center_y + 40, center_x - 30, center_y + 70), fill=(255, 180, 200))
        draw.ellipse((center_x + 30, center_y + 40, center_x + 60, center_y + 70), fill=(255, 180, 200))

# (remaining helper functions trimmed for brevity in script creation)
'''

raise SystemExit('incomplete script placeholder')

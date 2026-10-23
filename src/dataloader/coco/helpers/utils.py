from pathlib import Path

from PIL import Image


def read_image(image_path: Path, img_size: int) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize((img_size, img_size))


def read_labels(label_path: Path) -> list[list[float]]:
    """YOLO label file -> [[cls, cx, cy, w, h], ...]. A missing file means no objects."""
    if not label_path.exists():
        return []
    return [list(map(float, line.split())) for line in label_path.read_text().splitlines() if line.strip()]

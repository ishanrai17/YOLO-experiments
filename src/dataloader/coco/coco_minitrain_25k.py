from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from src.dataloader.coco.helpers.utils import read_image, read_labels


class CocoMinitrain25k(Dataset):
    """YOLO-format COCO -> (image, YOLOv1 target).

    Target is (S, S, 5 + C), each cell holding [x, y, w, h, obj, one-hot classes...],
    where x, y are offsets within the cell and w, h are relative to the image.
    """

    def __init__(self, root: str, split: str = "train", img_size: int = 448, S: int = 7, C: int = 80):
        root = Path(root)
        self.images = sorted((root / "images" / f"{split}2017").glob("*.jpg"))
        self.labels_dir = root / "labels" / f"{split}2017"
        self.img_size, self.S, self.C = img_size, S, C

        if not self.images:
            raise FileNotFoundError(f"No images in {root / 'images' / f'{split}2017'}, has the dataset been downloaded?")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        image = read_image(image_path, self.img_size)
        boxes = read_labels(self.labels_dir / f"{image_path.stem}.txt")
        return TF.to_tensor(image), self._encode(boxes)

    def _encode(self, boxes: list[list[float]]) -> torch.Tensor:
        S = self.S
        target = torch.zeros((S, S, 5 + self.C))
        best_area = torch.full((S, S), -1.0)

        # Single pass: the largest box in a cell wins, replacing any smaller one already there.
        for cls, cx, cy, w, h in boxes:
            i, j = min(int(cy * S), S - 1), min(int(cx * S), S - 1)
            if w * h <= best_area[i, j]:
                continue
            best_area[i, j] = w * h
            target[i, j] = 0
            target[i, j, :5] = torch.tensor([cx * S - j, cy * S - i, w, h, 1.0])
            target[i, j, 5 + int(cls)] = 1.0

        return target


def get_dataloader(root: str, split: str = "train", batch_size: int = 16, num_workers: int = 2, **kwargs) -> DataLoader:
    train = split == "train"
    return DataLoader(
        CocoMinitrain25k(root, split, **kwargs),
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
    )

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.dataloader.coco.coco_minitrain_25k import get_dataloader
from src.models.yolov1 import YoloV1


def yolo_loss(pred: torch.Tensor, target: torch.Tensor, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> torch.Tensor:
    """YOLOv1 sum-squared loss, one box per cell. pred predicts sqrt(w), sqrt(h); target holds w, h."""
    obj = target[..., 4]
    xy = ((pred[..., :2] - target[..., :2]) ** 2).sum(-1)
    wh = ((pred[..., 2:4] - target[..., 2:4].sqrt()) ** 2).sum(-1)
    conf = (pred[..., 4] - obj) ** 2
    cls = ((pred[..., 5:] - target[..., 5:]) ** 2).sum(-1)

    loss = lambda_coord * obj * (xy + wh) + obj * (conf + cls) + lambda_noobj * (1 - obj) * conf
    return loss.sum() / pred.shape[0]


def train(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    loader = get_dataloader(cfg.data_root, "train", cfg.batch_size, cfg.num_workers)
    model = YoloV1().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}")
        for step, (images, targets) in enumerate(pbar, 1):
            images, targets = images.to(device), targets.to(device)
            loss = yolo_loss(model(images), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{total / step:.4f}")

        torch.save(model.state_dict(), ckpt_dir / "last.pt")


@hydra.main(version_base=None, config_path="config", config_name="yolo_v0_coco_minitrain_25k")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()

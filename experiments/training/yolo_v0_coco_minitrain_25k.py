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


def save_checkpoint(ckpt_dir: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Save everything needed to resume training to ckpt_dir/epoch_{epoch + 1}.pt."""
    tmp = ckpt_dir / f"epoch_{epoch + 1}.tmp"
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(),  # keeps the shuffle order consistent after resuming
    }, tmp)
    tmp.replace(ckpt_dir / f"epoch_{epoch + 1}.pt")  # rename only once fully written, so resume never loads a partial file


def resume(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Load an epoch checkpoint into model and optimizer. Returns the epoch to start from."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng"])
    print(f"Resumed from {path}, starting at epoch {ckpt['epoch'] + 2}")
    return ckpt["epoch"] + 1

def load_checkpoints(ckpt_dir: Path) -> list[Path]:
    """Return a list of checkpoint paths in the directory, sorted by epoch number."""
    return sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """AdamW, with momentum as beta1. Configs without momentum / weight_decay get AdamW's defaults."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.get("momentum", 0.9), 0.999),
        weight_decay=cfg.get("weight_decay", 0.01),
        fused=cfg.get("fused_adam", False)
    )


def train(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    loader = get_dataloader(cfg.data_root, "train", cfg.batch_size, cfg.num_workers)
    model = YoloV1().to(device)
    optimizer = build_optimizer(cfg, model)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0

    if cfg.resume:
        ckpts = load_checkpoints(ckpt_dir)
        start_epoch = resume(ckpts[-1], model, optimizer, ckpt_dir) if ckpts else 0

    for epoch in range(start_epoch, cfg.epochs):
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

        save_checkpoint(ckpt_dir, epoch, model, optimizer)


@hydra.main(version_base=None, config_path="../config", config_name="yolo_v0_coco_minitrain_25k")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()

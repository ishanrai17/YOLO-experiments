import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from experiments.yolo.v0.training.coco_minitrain_25k import build_optimizer, resume, save_checkpoint, yolo_loss
from src.dataloader.coco.coco_minitrain_25k import CocoMinitrain25k
from src.models.yolov1 import YoloV1


def train(cfg: DictConfig) -> None:
    """fp32 DDP training, one process per GPU. Launch with torchrun --nproc_per_node=<num GPUs>."""
    # DDP: torchrun starts one process per GPU and passes each its rank through env vars
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    dataset = CocoMinitrain25k(cfg.data_root, "train")
    # DDP: each GPU gets a different 1/world_size slice of the data every epoch
    sampler = DistributedSampler(dataset, shuffle=True)
    # DDP: cfg.batch_size is the global batch, split evenly across GPUs
    loader = DataLoader(dataset, batch_size=cfg.batch_size // world_size, sampler=sampler,
                        drop_last=True, num_workers=cfg.num_workers)

    model = YoloV1().to(device)
    optimizer = build_optimizer(cfg, model)

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "last_epoch.pt"
    # DDP: every process loads the checkpoint, since each has its own optimizer
    start_epoch = resume(ckpt_path, model, optimizer) if cfg.resume and ckpt_path.exists() else 0
    # DDP: wrap after loading; from here on, backward averages gradients across GPUs
    model = DDP(model, device_ids=[local_rank])

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        # DDP: a new shuffle each epoch, identical on every GPU so their slices don't overlap
        sampler.set_epoch(epoch)
        total = 0.0
        # DDP: only rank 0 shows a progress bar, and its loss covers only its own slice of the data
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", disable=rank != 0)
        for step, (images, targets) in enumerate(pbar, 1):
            images, targets = images.to(device), targets.to(device)
            loss = yolo_loss(model(images), targets)

            optimizer.zero_grad()
            # DDP: backward also all-reduces, leaving the same averaged gradients on every GPU
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            optimizer.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{total / step:.4f}")

        # DDP: weights are identical on every GPU, so only rank 0 saves. model.module drops the DDP
        # wrapper, so the checkpoint also loads in the single-GPU script
        if rank == 0:
            save_checkpoint(ckpt_dir, epoch, model.module, optimizer)

    dist.destroy_process_group()


@hydra.main(version_base=None, config_path="../config", config_name="coco_minitrain_25k")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()

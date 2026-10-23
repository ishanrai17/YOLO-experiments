from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from experiments.training.yolo_v0_coco_minitrain_25k import build_optimizer, yolo_loss
from src.dataloader.coco.coco_minitrain_25k import get_dataloader
from src.models.yolov1 import YoloV1

WAIT, WARMUP, ACTIVE = 1, 2, 5  # profiler steps: skipped, run but not recorded, recorded


def profile_training(cfg: DictConfig) -> None:
    """Profile a few training steps, print the slowest ops, and save a trace for ui.perfetto.dev."""
    device = torch.device(cfg.device)
    loader = get_dataloader(cfg.data_root, "train", cfg.batch_size, cfg.num_workers)
    model = YoloV1().to(device).train()
    optimizer = build_optimizer(cfg, model)

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    batches = iter(loader)
    with profile(activities=activities, schedule=schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE)) as prof:
        for _ in range(WAIT + WARMUP + ACTIVE):
            with record_function("1_data"):
                images, targets = next(batches)
                images, targets = images.to(device), targets.to(device)
            with record_function("2_forward"):
                loss = yolo_loss(model(images), targets)
            with record_function("3_backward"):
                optimizer.zero_grad()
                loss.backward()
            with record_function("4_optimizer"):
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()  # finish this step's GPU work, so 1_data only measures waiting on the loader
            prof.step()

    sort_by = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=20))

    trace = Path(HydraConfig.get().runtime.output_dir) / "trace.json"
    prof.export_chrome_trace(str(trace))
    print(f"Trace saved to {trace}, open it at https://ui.perfetto.dev")


@hydra.main(version_base=None, config_path="../config", config_name="yolo_v0_coco_minitrain_25k")
def main(cfg: DictConfig) -> None:
    profile_training(cfg)


if __name__ == "__main__":
    main()

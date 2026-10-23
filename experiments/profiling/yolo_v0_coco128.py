import hydra
from omegaconf import DictConfig

from experiments.profiling.yolo_v0_coco_minitrain_25k import profile_training


@hydra.main(version_base=None, config_path="../config", config_name="yolo_v0_coco128")
def main(cfg: DictConfig) -> None:
    profile_training(cfg)


if __name__ == "__main__":
    main()

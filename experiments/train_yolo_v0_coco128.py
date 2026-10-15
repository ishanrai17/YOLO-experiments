import hydra
from omegaconf import DictConfig

from experiments.train_yolo_v0_coco_minitrain_25k import train


@hydra.main(version_base=None, config_path="config", config_name="yolo_v0_coco128")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()

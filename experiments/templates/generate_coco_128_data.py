import hydra
from omegaconf import DictConfig

from src.data.coco.extract.extractor_v1 import ExtractorV1

@hydra.main(version_base=None, config_path="../config/data", config_name="coco128")
def run(cfg: DictConfig) -> None:
    extractor = ExtractorV1(cfg.url, cfg.identifier)
    folder_path = extractor.extract()
    print(f"Data extracted to: {folder_path}")


if __name__ == "__main__":
    run()
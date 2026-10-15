import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.data.coco.extract.helpers.utils import download_and_unzip

class Coco128:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def download(self) -> str:
        return download_and_unzip(
            url=self.cfg.url,
            folder_path=Path(self.cfg.folder_path)
        )

@hydra.main(version_base=None, config_path="config", config_name="coco128")
def main(cfg: DictConfig) -> None:
    folder_path = Coco128(cfg).download()
    print(f"Data extracted to: {folder_path}")


if __name__ == "__main__":
    main()

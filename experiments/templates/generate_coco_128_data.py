import hydra

from ...src.data.coco.extract.extractor_v1 import ExtractorV1

@hydra.main(config_path="../../../../experiments/config/data", config_name="coco128")
def run(cfg):
    url = cfg.url
    identifier = cfg.identifier

    extractor = ExtractorV1(url, identifier)
    folder_path = extractor.extract()
    print(f"Data extracted to: {folder_path}")
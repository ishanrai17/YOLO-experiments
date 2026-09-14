from ..extract.extractor_v1 import ExtractorV1


class COCO128Extractor(ExtractorV1):
    def __init__(self):
        super().__init__(
            url="https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
            identifier="coco128",
        )

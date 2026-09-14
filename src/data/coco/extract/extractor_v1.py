import urllib.request, zipfile
from pathlib import Path

from .abstract_extractor import AbstractExtractor

class ExtractorV1(AbstractExtractor):
    def __init__(self, url: str, folder_path: Path):
        self.url = url
        self.folder_path = folder_path

    def extract(self):
        self.folder_path.mkdir(parents=True, exist_ok=True)
        archive_path = self.folder_path / self.url.rsplit("/", 1)[-1]

        if not archive_path.exists():
            urllib.request.urlretrieve(self.url, archive_path)

        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as z:
                z.extractall(self.folder_path / "raw")

        return str(self.folder_path)
import wget, zipfile
from pathlib import Path


def download_and_unzip(url: str, folder_path: Path) -> str:
    folder_path.mkdir(parents=True, exist_ok=True)
    archive_path = folder_path / url.rsplit("/", 1)[-1]

    if not archive_path.exists():
        wget.download(url, out=str(archive_path))
        print()

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(folder_path / "raw")

    return str(folder_path)

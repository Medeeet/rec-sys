"""Download Amazon Electronics 2018 review + metadata files."""

import gzip
import json
import urllib.request
from pathlib import Path

from src.utils import get_logger

logger = get_logger(__name__)


def download_file(url: str, dest: Path) -> Path:
    """Download a file if it doesn't exist yet."""
    if dest.exists():
        logger.info(f"File already exists: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, str(dest))
    logger.info(f"Downloaded: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def parse_json_gz(filepath: Path) -> list[dict]:
    """Parse gzipped JSON-lines file."""
    records = []
    logger.info(f"Parsing {filepath}...")
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    logger.info(f"Parsed {len(records)} records from {filepath.name}")
    return records


def download_amazon_electronics(raw_dir: str = "data/raw") -> tuple[Path, Path]:
    """Download Electronics 5-core reviews and metadata.

    Returns paths to the downloaded .json.gz files.
    """
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    reviews_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Electronics_5.json.gz"
    metadata_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Electronics.json.gz"

    reviews_path = download_file(reviews_url, raw_path / "Electronics_5.json.gz")
    metadata_path = download_file(metadata_url, raw_path / "meta_Electronics.json.gz")

    return reviews_path, metadata_path


if __name__ == "__main__":
    download_amazon_electronics()

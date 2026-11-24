import argparse
from pathlib import Path
from zipfile import ZipFile

import gdown

from src.utils import sha256


DATA_ID = "10f1H2T-5W-BiqabHHtlZ4ASs19TZmg8R"
ARCHIVE_SHA256 = "7ba4eb8dae0352ee53e583166d9f7c9838a93fc3ecca936624af9089c59bfdc8"


def download(destination, archive_path):
    destination, archive_path = Path(destination).resolve(), Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        gdown.download(id=DATA_ID, output=str(archive_path), quiet=False)
    if sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA256 mismatch; remove it and download the original archive again")
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError("Unsafe archive path")
        archive.extractall(destination)
    print(f"Data root: {destination / 'data'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default=".")
    parser.add_argument("--archive", default=".runtime/data.zip")
    args = parser.parse_args()
    download(args.destination, args.archive)

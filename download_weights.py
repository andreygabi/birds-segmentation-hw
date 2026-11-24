import argparse
import json
from pathlib import Path
import urllib.request

from src.utils import sha256


def download(output, include_onnx):
    manifest = json.loads((Path(__file__).parent / "results/weights_manifest.json").read_text())
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    for artifact in manifest["artifacts"]:
        if artifact["name"].endswith(".onnx") and not include_onnx:
            continue
        path = root / artifact["name"]
        if path.exists() and sha256(path) == artifact["sha256"]:
            print(f"Verified {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        urllib.request.urlretrieve(artifact["url"], temporary)
        if sha256(temporary) != artifact["sha256"]:
            temporary.unlink()
            raise ValueError(f"SHA256 mismatch: {artifact['name']}")
        temporary.replace(path)
        print(f"Downloaded and verified {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights")
    parser.add_argument("--onnx", action="store_true")
    args = parser.parse_args()
    download(args.output, args.onnx)

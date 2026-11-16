"""Download the multilingual model from ModelScope."""

import argparse
import os
import time
from pathlib import Path

from requests.exceptions import ConnectionError, SSLError, Timeout


REPO = "convaiinnovations/laya-multilingual"
FILES = (
    "rl_agent_config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
)


def download_model(root: Path, *, if_missing: bool = False) -> None:
    destination = root / "multilingual"
    if if_missing and all(
        (destination / filename).is_file() and (destination / filename).stat().st_size > 0
        for filename in FILES
    ):
        print(f"Multilingual model already exists in {destination}; skipping download", flush=True)
        return

    from modelscope import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO} from ModelScope", flush=True)
    for attempt in range(5):
        try:
            snapshot_download(
                model_id=REPO, revision="master", local_dir=str(destination),
                allow_patterns=list(FILES),
            )
            break
        except (ConnectionError, SSLError, Timeout):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    for filename in FILES:
        path = destination / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"ModelScope snapshot is missing required model file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-missing", action="store_true", help="Skip downloading when all required model files exist")
    args = parser.parse_args()
    root = Path(os.environ.get("LAYA_MODEL_DIR", "models"))
    root.mkdir(parents=True, exist_ok=True)
    download_model(root, if_missing=args.if_missing)
    print(f"Multilingual model is ready in {root / 'multilingual'}")


if __name__ == "__main__":
    main()

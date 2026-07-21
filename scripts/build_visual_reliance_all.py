"""Collate completed visual-reliance model JSONs into one aggregate."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    combined = {}
    for path in sorted(args.directory.glob("*.json")):
        if path.name == "visual_reliance_all.json":
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        model = result.get("model")
        if model and result.get("levels"):
            combined[model] = result

    output = args.directory / "visual_reliance_all.json"
    temp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2)
    os.replace(temp, output)
    print(f"Wrote {output} ({len(combined)} models)")


if __name__ == "__main__":
    main()

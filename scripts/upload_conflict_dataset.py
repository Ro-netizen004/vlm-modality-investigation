#!/usr/bin/env python3
"""
Upload the built conflict dataset to the Hugging Face Hub (vlm-modality-research org).

Requires an HF token with WRITE access to the org. Authenticate first (in your
own terminal, so the token is never passed through anything else):

    huggingface-cli login
    #   or:  export HF_TOKEN=hf_xxx   (Windows PowerShell: $env:HF_TOKEN="hf_xxx")

Then:

    # private first (recommended — review on the Hub before making public)
    python scripts/upload_conflict_dataset.py

    # make public
    python scripts/upload_conflict_dataset.py --public

    # override repo name / local source
    python scripts/upload_conflict_dataset.py \
        --repo vlm-modality-research/modality-conflict-arbitration-v1 \
        --data-dir data/conflict_dataset_v1
"""
import argparse
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi, whoami

DEFAULT_REPO = "vlm-modality-research/modality-conflict-arbitration-v1"
DEFAULT_DATA = "data/conflict_dataset_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--public", action="store_true",
                    help="create/set the repo public (default: private)")
    args = ap.parse_args()

    # Fail fast with a clear message if not authenticated.
    try:
        me = whoami()
    except Exception:
        raise SystemExit("Not authenticated. Run `huggingface-cli login` or set "
                         "HF_TOKEN, then re-run.")
    org = args.repo.split("/")[0]
    orgs = [o.get("name") for o in me.get("orgs", [])]
    print(f"Authenticated as: {me.get('name')}  | orgs: {orgs}")
    if org not in orgs and me.get("name") != org:
        print(f"WARNING: '{org}' is not in your org list — push will fail without "
              f"write access to it.")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"No dataset at {data_dir}. Build it first: "
                         f"python scripts/build_conflict_dataset.py")

    ds = load_from_disk(str(data_dir))
    print(f"Loaded {len(ds)} rows from {data_dir}")
    private = not args.public
    print(f"Pushing to {args.repo}  (private={private}) ...")
    ds.push_to_hub(args.repo, private=private)

    # Upload the dataset card as the repo README.
    card = data_dir / "README.md"
    if card.exists():
        HfApi().upload_file(
            path_or_fileobj=str(card),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
        )
        print("Uploaded dataset card (README.md).")
    print(f"Done: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

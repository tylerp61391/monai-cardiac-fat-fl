"""Script to automate uploading ONNX model weights to GitHub Releases via REST API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY") or "tylerp61391/monai-cardiac-fat-fl"
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1.0.0"

    repo_root = Path(__file__).resolve().parents[1]
    model_dir = repo_root / "models"
    files_to_upload = [
        model_dir / "deepheartseg_loc.onnx",
        model_dir / "deepheartseg_heart.onnx",
    ]

    for f in files_to_upload:
        if not f.is_file():
            print(f"Error: Required model file not found: {f}")
            sys.exit(1)

    if not token:
        print("[!] GITHUB_TOKEN environment variable not detected.")
        print("    To publish programmatically via terminal or an autonomous agent:")
        print("    $env:GITHUB_TOKEN = '<your-github-personal-access-token>'")
        print("    python scripts/publish_release.py [tag]")
        print("\n    Alternatively, follow the web drag-and-drop steps in MODEL_RELEASE_INSTRUCTIONS.md.")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1. Check or create release
    rel_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    res = requests.get(rel_url, headers=headers)
    if res.status_code == 200:
        release_id = res.json()["id"]
        upload_url = res.json()["upload_url"].split("{")[0]
        print(f"[*] Found existing release {tag} (ID: {release_id})")
    else:
        create_url = f"https://api.github.com/repos/{repo}/releases"
        payload = {
            "tag_name": tag,
            "target_commitish": "master",
            "name": f"{tag} - Production MONAI & ONNX Pretrained Models",
            "body": "Pretrained DeepHeartSeg ONNX models matching foundational MGH CardiacFatQuant with 0.00 delta.",
            "draft": False,
            "prerelease": False,
        }
        create_res = requests.post(create_url, headers=headers, json=payload)
        create_res.raise_for_status()
        upload_url = create_res.json()["upload_url"].split("{")[0]
        print(f"[+] Created release {tag}")

    # 2. Upload assets
    for f in files_to_upload:
        asset_url = f"{upload_url}?name={f.name}"
        print(f"[*] Uploading {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)...")
        upload_headers = dict(headers)
        upload_headers["Content-Type"] = "application/octet-stream"
        with open(f, "rb") as fp:
            up_res = requests.post(asset_url, headers=upload_headers, data=fp)
            if up_res.status_code in (201, 200):
                print(f"[+] Successfully uploaded {f.name}")
            else:
                print(f"[-] Failed to upload {f.name}: {up_res.status_code} - {up_res.text}")


if __name__ == "__main__":
    main()


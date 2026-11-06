"""Download NVIDIA NV-Reason-CT weights and tokenizer to local storage.

Downloads all repository files from Hugging Face (nvidia/NV-Reason-CT)
to local directory E:\\models\\NV-Reason-CT with zero GPU VRAM impact.
"""

from pathlib import Path
from huggingface_hub import snapshot_download


def download_model(target_dir: Path = Path(r"E:\models\NV-Reason-CT")) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting background download of nvidia/NV-Reason-CT to: {target_dir}")
    print("This runs entirely on CPU and disk I/O (0 MB VRAM used).")

    downloaded_path = snapshot_download(
        repo_id="nvidia/NV-Reason-CT",
        local_dir=str(target_dir),
    )
    print(f"\nDownload completed successfully! All assets saved to: {downloaded_path}")
    return Path(downloaded_path)


if __name__ == "__main__":
    download_model()

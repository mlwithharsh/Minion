"""
Minion AI — inference/sync_checkpoint.py
Download the latest Minion checkpoint from Google Drive to your local machine.

This is the "trained on Colab, run locally" handoff step.
The Colab training script saves checkpoints to Google Drive.
This script fetches the latest checkpoint to your local machine.

Two sync methods:
  Method A (recommended): rclone — fast, resumable, supports full Drive sync.
      Requires one-time setup: rclone config (follow interactive prompts).
      https://rclone.org/drive/

  Method B: gdown — simple Python-only download from a shareable Drive link.
      Requires a Drive folder shareable link (anyone with link can view).
      No rclone needed, but less reliable for large files.

  Method C: Manual — if Drive is mounted locally (e.g. via Google Drive app),
      just specify the local mount path.

Usage:
    # Method A (rclone):
    python inference/sync_checkpoint.py \\
        --method rclone \\
        --remote "gdrive:minion_ckpts" \\
        --local_dir ./checkpoints

    # Method B (gdown, shareable link):
    python inference/sync_checkpoint.py \\
        --method gdown \\
        --drive_folder_url "https://drive.google.com/drive/folders/FOLDER_ID" \\
        --local_dir ./checkpoints

    # Method C (local mount):
    python inference/sync_checkpoint.py \\
        --method local \\
        --drive_mount "/mnt/google-drive/My Drive/minion_ckpts" \\
        --local_dir ./checkpoints
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Sync methods
# ---------------------------------------------------------------------------

def sync_rclone(remote: str, local_dir: str, dry_run: bool = False) -> None:
    """Sync Drive folder to local using rclone.

    Setup (one time):
        rclone config        (creates ~/.config/rclone/rclone.conf)
        Choose Google Drive as remote, name it 'gdrive'

    Args:
        remote:    rclone remote path, e.g. 'gdrive:minion_ckpts'
        local_dir: Local destination directory
        dry_run:   If True, print what would be downloaded without downloading
    """
    if shutil.which("rclone") is None:
        print(
            "rclone not found.  Install with:\n"
            "  Windows: winget install Rclone.Rclone\n"
            "  Linux:   sudo apt install rclone\n"
            "  Mac:     brew install rclone\n"
            "Then run: rclone config  (to set up Google Drive remote)"
        )
        sys.exit(1)

    os.makedirs(local_dir, exist_ok=True)

    cmd = ["rclone", "sync", remote, local_dir, "--progress"]
    if dry_run:
        cmd.append("--dry-run")

    print(f"Syncing {remote} → {local_dir}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"rclone exited with code {result.returncode}")
        sys.exit(result.returncode)

    _report_latest(local_dir)


def sync_gdown(
    drive_folder_url: str,
    local_dir: str,
    filename: str = "ckpt.pt",
) -> None:
    """Download the latest checkpoint via gdown from a shareable Drive link.

    Args:
        drive_folder_url: Shareable Google Drive folder URL
                          (anyone with link → viewer access required)
        local_dir:        Local destination directory
        filename:         Name of the checkpoint file to look for

    Note: gdown can be unreliable for large files.  For production use,
    prefer rclone (Method A).
    """
    try:
        import gdown
    except ImportError:
        print("gdown not installed. Run: pip install gdown")
        sys.exit(1)

    os.makedirs(local_dir, exist_ok=True)
    output = os.path.join(local_dir, filename)

    print(f"Downloading from {drive_folder_url}")
    print("(Make sure the Drive folder is set to 'Anyone with link can view')")

    # Extract folder ID from URL
    if "folders/" in drive_folder_url:
        folder_id = drive_folder_url.split("folders/")[1].split("?")[0]
    else:
        folder_id = drive_folder_url

    # Download all files from the folder
    try:
        gdown.download_folder(
            url         = f"https://drive.google.com/drive/folders/{folder_id}",
            output      = local_dir,
            quiet       = False,
            use_cookies = False,
        )
    except Exception as e:
        print(f"gdown failed: {e}")
        print(
            "Troubleshooting:\n"
            "  1. Ensure the Drive folder is shared with 'Anyone with link → Viewer'\n"
            "  2. Try gdown manually: gdown --folder <url> -O ./checkpoints\n"
            "  3. For large files (>1 GB), gdown may hit rate limits — use rclone instead"
        )
        sys.exit(1)

    _report_latest(local_dir)


def sync_local_mount(drive_mount: str, local_dir: str) -> None:
    """Copy checkpoints from a locally mounted Drive path.

    Use this if Google Drive is mounted on your machine via the
    Google Drive desktop application.

    Args:
        drive_mount: Path to the Drive folder on your mounted filesystem
                     e.g. "C:/Users/Me/Google Drive/minion_ckpts"
        local_dir:   Local destination directory
    """
    src = Path(drive_mount)
    if not src.exists():
        print(f"Drive mount path not found: {src}")
        print(
            "Make sure Google Drive is installed and synced, or use --method rclone."
        )
        sys.exit(1)

    dst = Path(local_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Copy all .pt files
    copied = 0
    for pt_file in src.glob("*.pt"):
        shutil.copy2(pt_file, dst / pt_file.name)
        print(f"Copied: {pt_file.name}  ({pt_file.stat().st_size / 1024**2:.1f} MB)")
        copied += 1

    print(f"Copied {copied} checkpoint(s) to {dst}")
    _report_latest(str(dst))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _report_latest(local_dir: str) -> None:
    """Print info about the most recently modified checkpoint in local_dir."""
    ckpts = list(Path(local_dir).glob("*.pt"))
    if not ckpts:
        print(f"No .pt checkpoints found in {local_dir}")
        return

    latest = max(ckpts, key=lambda p: p.stat().st_mtime)
    size_mb = latest.stat().st_size / 1024 ** 2
    print(f"\nLatest checkpoint: {latest.name}  ({size_mb:.1f} MB)")
    print(f"Full path: {latest.resolve()}")
    print(f"\nTo generate from this checkpoint:")
    print(f"  python inference/generate.py --checkpoint {latest.resolve()} --prompt 'Hello'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync Minion AI checkpoint from Google Drive to local machine"
    )
    parser.add_argument("--method", required=True,
                        choices=["rclone", "gdown", "local"],
                        help="Sync method to use")

    # rclone args
    parser.add_argument("--remote", default="gdrive:minion_ckpts",
                        help="rclone remote path (method=rclone)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be downloaded (method=rclone)")

    # gdown args
    parser.add_argument("--drive_folder_url", default=None,
                        help="Shareable Drive folder URL (method=gdown)")

    # local mount args
    parser.add_argument("--drive_mount", default=None,
                        help="Local path to mounted Drive folder (method=local)")

    # common
    parser.add_argument("--local_dir", default="./checkpoints",
                        help="Local directory to download checkpoints into")

    args = parser.parse_args()

    if args.method == "rclone":
        sync_rclone(args.remote, args.local_dir, dry_run=args.dry_run)

    elif args.method == "gdown":
        if not args.drive_folder_url:
            parser.error("--drive_folder_url required for --method gdown")
        sync_gdown(args.drive_folder_url, args.local_dir)

    elif args.method == "local":
        if not args.drive_mount:
            parser.error("--drive_mount required for --method local")
        sync_local_mount(args.drive_mount, args.local_dir)


if __name__ == "__main__":
    main()

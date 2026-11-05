"""Where a result comes from: the version of the code and of the data."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def git_commit() -> str:
    """Hash of the current Git commit, or 'unknown' outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def dvc_md5(data_file: Path) -> str:
    """MD5 recorded by DVC for `data_file`, read from its `.dvc` pointer file.

    This is the identifier of the data version: `dvc checkout` with the same
    pointer file restores exactly these bytes.
    """
    pointer = data_file.with_name(data_file.name + ".dvc")
    if not pointer.exists():
        return "untracked"
    with open(pointer, encoding="utf-8") as f:
        return str(yaml.safe_load(f)["outs"][0]["md5"])

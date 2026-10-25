import os
import subprocess

from aerotrack.paths import REPO_ROOT


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_info() -> dict:
    """Commit + dirty flag. A dirty run also stores its diff, otherwise the commit alone
    does not identify the code that produced the numbers."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None and os.environ.get("AEROTRACK_GIT_COMMIT"):
        # Inside the training image there is no .git; the commit was baked in at build time.
        dirty = os.environ.get("AEROTRACK_GIT_DIRTY")
        return {"commit": os.environ["AEROTRACK_GIT_COMMIT"], "branch": None,
                "dirty": {"true": True, "false": False}.get(dirty), "source": "image build arg"}
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def git_diff() -> str:
    return _git("diff", "HEAD") or ""

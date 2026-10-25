"""Archive download + extraction into data/raw/, which is never modified afterwards.

Each source declares a sha256. Extraction writes a `.complete` marker so re-running
`make data` is a no-op, and the extracted files are made read-only: converters can only
read raw data, and an accidental write fails loudly instead of corrupting it.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

import requests
from pydantic import BaseModel

from aerotrack.paths import ARCHIVES_DIR, RAW_DIR


class Source(BaseModel):
    filename: str
    url: str | None = None
    gdrive_id: str | None = None
    sha256: str | None = None
    official_page: str | None = None
    # Directory, relative to the dataset's raw dir, the archive is extracted into.
    extract_to: str = "."


class ManualDownloadRequired(RuntimeError):
    pass


def sha256sum(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _http_download(url: str, dest: Path, params: dict | None = None,
                   session: requests.Session | None = None) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"Range": f"bytes={tmp.stat().st_size}-"} if tmp.exists() else {}
    get = (session or requests).get
    with get(url, params=params, stream=True, timeout=60, headers=headers) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("content-type", ""):
            # Drive answers quota errors with a 200 HTML page; saving it under the archive's
            # name would only fail later, at extraction, with a misleading error.
            title = re.search(r"<title>(.*?)</title>", r.text, flags=re.S)
            raise ManualDownloadRequired(f"server returned a web page ({title.group(1).strip()
                                         if title else 'no title'}) instead of the archive")
        mode = "ab" if r.status_code == 206 else "wb"
        total = int(r.headers.get("content-length", 0))
        done, last = 0, -1
        with open(tmp, mode) as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
                done += len(block)
                pct = int(100 * done / total) if total else -1
                if pct != last and pct % 10 == 0:
                    print(f"  {dest.name}: {pct}%", flush=True)
                    last = pct
    tmp.rename(dest)


def _gdrive_download(file_id: str, dest: Path) -> None:
    """Large Drive files answer with a "virus scan warning" page whose form carries the real
    download parameters. gdown handles this through cached cookies, which proved brittle
    (it failed on the VisDrone-MOT archives), so the form is followed explicitly."""
    session = requests.Session()
    r = session.get("https://drive.google.com/uc", params={"export": "download", "id": file_id},
                    timeout=60)
    r.raise_for_status()
    if "text/html" not in r.headers.get("content-type", ""):
        dest.write_bytes(r.content)  # small file served directly
        return
    form = re.search(r'<form id="download-form" action="([^"]+)"', r.text)
    if not form:
        raise ManualDownloadRequired(f"{dest.name}: no download form (quota exceeded or private)")
    params = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', r.text))
    _http_download(html.unescape(form.group(1)), dest, params=params, session=session)


def fetch(src: Source) -> Path:
    """Return the verified local archive, downloading it if absent."""
    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVES_DIR / src.filename
    if not dest.exists():
        print(f"downloading {src.filename}", flush=True)
        try:
            if src.url:
                _http_download(src.url, dest)
            elif src.gdrive_id:
                _gdrive_download(src.gdrive_id, dest)
            else:
                raise ManualDownloadRequired(src.filename)
        except (ManualDownloadRequired, requests.RequestException, OSError) as e:
            link = src.url or (f"https://drive.google.com/file/d/{src.gdrive_id}"
                               if src.gdrive_id else src.official_page)
            raise ManualDownloadRequired(
                f"Could not download {src.filename} automatically ({e}).\n"
                f"Download it manually from {link}\n"
                f"and place it at {dest}, then re-run `make data`."
            ) from e
    if src.sha256:
        digest = sha256sum(dest)
        if digest != src.sha256:
            raise RuntimeError(
                f"sha256 mismatch for {dest}: got {digest}, expected {src.sha256}. "
                "Delete the file and re-download."
            )
    else:
        print(f"  warning: no sha256 pinned for {src.filename}: {sha256sum(dest)}", flush=True)
    return dest


def _extract(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    elif archive.name.endswith((".tar", ".tar.gz", ".tgz")):
        shutil.unpack_archive(str(archive), str(target))
    elif archive.suffix in {".rar", ".7z"}:
        # Some UAVDT archives are not zip; bsdtar (macOS default tar, libarchive on Linux)
        # reads them without an extra Python dependency.
        subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(target)], check=True)
    else:
        raise ValueError(f"unsupported archive format: {archive}")


def make_read_only(root: Path) -> None:
    """Files only: directories stay writable so `rm -rf data/raw/x` still works."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def ensure_raw(dataset: str, sources: list[Source], keep_archives: bool = True) -> Path:
    raw = RAW_DIR / dataset
    for src in sources:
        marker = raw / f".{src.filename}.complete"
        if marker.exists():
            continue
        archive = fetch(src)
        target = raw / src.extract_to
        print(f"extracting {archive.name} -> {target}", flush=True)
        _extract(archive, target)
        make_read_only(target)
        marker.write_text(json.dumps({"sha256": src.sha256 or sha256sum(archive)}))
        if not keep_archives:
            archive.unlink()
    return raw

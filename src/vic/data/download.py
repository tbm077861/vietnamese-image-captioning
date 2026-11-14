"""Resumable file downloads (plain HTTP or Google Drive) and safe zip extraction."""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

import requests

GDRIVE_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
CHUNK_BYTES = 1 << 20


def resolve_url(file_spec: dict) -> str:
    if "url" in file_spec:
        return file_spec["url"]
    if "gdrive_id" in file_spec:
        return GDRIVE_URL.format(file_id=file_spec["gdrive_id"])
    raise ValueError(f"File entry {file_spec.get('name')!r} needs either 'url' or 'gdrive_id'")


def download_file(
    url: str, destination: Path, expected_bytes: int | None, logger: logging.Logger
) -> Path:
    """Download ``url`` to ``destination``; an interrupted download resumes on the next call."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_bytes in (None, destination.stat().st_size):
        logger.info(f"{destination.name}: already downloaded")
        return destination

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_bytes is not None and offset > expected_bytes:
        partial.unlink()
        offset = 0
    if expected_bytes is None or offset < expected_bytes:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(url, headers=headers, stream=True, timeout=60) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                raise RuntimeError(
                    f"{destination.name}: the server returned a web page instead of the file "
                    "(download quota exceeded or the link changed). Download it manually from "
                    f"the dataset homepage and save it as {destination}."
                )
            if offset and response.status_code != 206:
                offset = 0  # the server ignored the range request: start over
            total = offset + int(response.headers.get("Content-Length", 0))
            done = offset
            next_report = done + 250 * CHUNK_BYTES
            logger.info(f"{destination.name}: downloading {total / 1e6:.1f} MB")
            with open(partial, "ab" if offset else "wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    handle.write(chunk)
                    done += len(chunk)
                    if done >= next_report:
                        logger.info(f"{destination.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
                        next_report += 250 * CHUNK_BYTES

    size = partial.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise RuntimeError(
            f"{destination.name}: got {size} bytes, expected {expected_bytes}. "
            "Run the command again to resume."
        )
    os.replace(partial, destination)
    return destination


def extract_zip(archive: Path, target_dir: Path, logger: logging.Logger) -> None:
    """Extract ``archive`` into ``target_dir``, refusing entries that escape the directory."""
    target_root = target_dir.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (target_dir / member.filename).resolve().is_relative_to(target_root):
                raise RuntimeError(f"Unsafe path in {archive.name}: {member.filename}")
        logger.info(f"{archive.name}: extracting {len(bundle.infolist())} entries")
        bundle.extractall(target_dir)


def extraction_marker(target_dir: Path, archive_name: str) -> Path:
    """Marker written after a successful extraction, so the archive is not needed again."""
    return target_dir / f".{archive_name}.extracted"

#!/usr/bin/env python3
"""Prepare the official NW-UCLA RGB package entirely inside a Kaggle kernel."""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OFFICIAL_PAGE = "https://wangjiangb.github.io/my_data.html"
USERNAME = "vichanshur"
DATASET_SLUG = f"{USERNAME}/northwestern-ucla-multiview-action-rgb-base-paper"
ROOT = Path("/kaggle/working/nwucla_setup")
DOWNLOADS, EXTRACTED, PUBLISH = ROOT / "downloads", ROOT / "extracted", ROOT / "publish"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}
RELEVANT_EXTS = VIDEO_EXTS | {".jpg", ".jpeg", ".png"}


def run(args, check=True):
    print("$", " ".join(map(str, args)), flush=True)
    return subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check).stdout


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB": return f"{n:.2f} {unit}"
        n /= 1024


def disk_usage(path):
    u = shutil.disk_usage(path)
    return f"{human(u.free)} free of {human(u.total)}"


def kaggle_guard():
    print("CWD=", os.getcwd())
    print("KAGGLE_INPUT_EXISTS=", Path("/kaggle/input").is_dir())
    print("KAGGLE_WORKING_EXISTS=", Path("/kaggle/working").is_dir())
    print("FREE_DISK=", disk_usage("/kaggle/working" if Path("/kaggle/working").is_dir() else "/"))
    print("PYTHON_VERSION=", sys.version.replace("\n", " "))
    if not Path("/kaggle/working").is_dir():
        raise SystemExit("ERROR: THIS SCRIPT IS NOT RUNNING ON KAGGLE.")


def session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Kaggle NW-UCLA dataset preparation)"})
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "HEAD"))))
    return s


def find_rgb_url(s):
    r = s.get(OFFICIAL_PAGE, timeout=(20, 60)); r.raise_for_status()
    page = r.text
    print(f"OFFICIAL_DATASET_PAGE={OFFICIAL_PAGE}")
    # Find an anchor whose local textual context is both the dataset and RGB videos only.
    anchors = re.findall(r"<a\b[^>]*?href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", page, flags=re.I | re.S)
    candidates = []
    for href, label in anchors:
        label = re.sub(r"<[^>]+>", " ", html.unescape(label))
        pos = page.lower().find(href.lower())
        context = page[max(0, pos - 3000):pos + 1000].lower() if pos >= 0 else ""
        if "rgb videos only" in label.lower() or ("northwestern-ucla" in context and "rgb" in label.lower()):
            candidates.append((href, label.strip()))
    if not candidates:
        raise RuntimeError("Official page did not expose an RGB-only NW-UCLA link; refusing to guess a URL.")
    href, label = candidates[0]
    url = urljoin(OFFICIAL_PAGE, href)
    print("RGB_LINK_TEXT=", label)
    print(f"RESOLVED_RGB_DOWNLOAD_URL={url}")
    return url


def filename_from_response(response, url):
    cd = response.headers.get("Content-Disposition", "")
    match = re.search(r"filename[*]?=(?:UTF-8'')?[\"']?([^;\"']+)", cd, re.I)
    name = match.group(1) if match else url.rstrip("/").split("/")[-1].split("?")[0]
    return re.sub(r"[^A-Za-z0-9._-]", "_", name or "nwucla_rgb_download")


def download(s, url):
    probe = s.get(url, stream=True, allow_redirects=True, timeout=(30, 120))
    probe.raise_for_status()
    name = filename_from_response(probe, probe.url)
    target = DOWNLOADS / name
    # A response which resolves to HTML is an access/landing page, never dataset content.
    content_type = probe.headers.get("Content-Type", "")
    if "text/html" in content_type.lower():
        raise RuntimeError(f"Resolved URL returned HTML ({probe.url}); direct official download is unavailable.")
    expected = int(probe.headers.get("Content-Length", "0") or 0)
    existing = target.stat().st_size if target.exists() else 0
    probe.close()
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    r = s.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(30, 300))
    if existing and r.status_code != 206:
        existing = 0
        target.unlink(missing_ok=True)
        r.close(); r = s.get(url, stream=True, allow_redirects=True, timeout=(30, 300))
    r.raise_for_status()
    mode = "ab" if existing else "wb"
    downloaded = existing
    with target.open(mode) as f:
        for chunk in r.iter_content(1024 * 1024):
            if chunk:
                f.write(chunk); downloaded += len(chunk)
                if downloaded % (64 * 1024 * 1024) < len(chunk): print(f"DOWNLOAD_PROGRESS={human(downloaded)}", flush=True)
    print(f"DOWNLOAD_FILENAME={target.name}")
    print(f"DOWNLOAD_HTTP_STATUS={r.status_code}")
    print(f"DOWNLOAD_CONTENT_TYPE={r.headers.get('Content-Type', '')}")
    print(f"DOWNLOAD_SIZE_BYTES={target.stat().st_size}")
    print(f"DOWNLOAD_SIZE_HUMAN={human(target.stat().st_size)}")
    if target.stat().st_size < 1024 * 1024: raise RuntimeError("Downloaded object is implausibly small for RGB videos.")
    with target.open("rb") as f: magic = f.read(16)
    if magic.lstrip().lower().startswith(b"<!doctype") or magic.lstrip().lower().startswith(b"<html"):
        raise RuntimeError("Downloaded HTML instead of dataset content.")
    print("FILE_TYPE=", run(["file", "-b", str(target)], check=False).strip())
    return target


def extract(archive):
    low = archive.name.lower()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad: raise RuntimeError(f"ZIP integrity failure: {bad}")
            z.extractall(EXTRACTED)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(EXTRACTED, filter="data")
    elif low.endswith((".7z", ".rar", ".zip.001")):
        print(run(["7z", "l", str(archive)]))
        print(run(["7z", "x", "-y", f"-o{EXTRACTED}", str(archive)]))
    else:
        raise RuntimeError(f"Unsupported/unrecognized archive format: {archive.name}")


def validate():
    files = [p for p in EXTRACTED.rglob("*") if p.is_file()]
    relevant = [p for p in files if p.suffix.lower() in RELEVANT_EXTS]
    videos = [p for p in files if p.suffix.lower() in VIDEO_EXTS]
    ext_counts = {e: sum(p.suffix.lower() == e for p in files) for e in sorted(RELEVANT_EXTS)}
    names = "\n".join(str(p.relative_to(EXTRACTED)) for p in relevant[:50])
    print("FIRST_50_RELEVANT_PATHS:\n" + names)
    print("EXTENSION_COUNTS=", json.dumps(ext_counts, sort_keys=True))
    size = sum(p.stat().st_size for p in files)
    all_names = " ".join(str(p).lower() for p in files[:10000])
    likely = bool(relevant) and ("ucla" in all_names or "view" in all_names or "a01" in all_names or "s01" in all_names)
    status = "PASS" if likely else ("WARNING" if relevant else "FAIL")
    print("VALIDATION_DATASET_NAME=Northwestern-UCLA Multiview Action 3D" if likely else "VALIDATION_DATASET_NAME=UNKNOWN")
    print(f"VALIDATION_TOTAL_FILES={len(files)}")
    print(f"VALIDATION_VIDEO_FILES={len(videos)}")
    print(f"VALIDATION_SIZE={human(size)}")
    print(f"VALIDATION_STATUS={status}")
    if status == "FAIL": raise RuntimeError("Content clearly does not contain NW-UCLA RGB files; will not publish.")
    return files, videos, size, status


def publish(files, videos, size, status):
    dataset_root = PUBLISH / "NW-UCLA"
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    # Move rather than duplicate once validation has finished.
    shutil.move(str(EXTRACTED), str(dataset_root))
    (PUBLISH / "README.md").write_text("""# Northwestern-UCLA Multiview Action 3D

Official source: https://wangjiangb.github.io/my_data.html

Purpose: reproduction of *Multi-view Human Action Recognition Based on TSN Architecture Integrated with GRU*.

Modality: RGB video. Expected characteristics: 10 actions, 10 actors, 3 views.

This is a private reusable copy prepared from the original research dataset source. Original dataset usage terms are governed by the dataset authors/source.
""")
    (PUBLISH / "dataset_info.txt").write_text(f"source={OFFICIAL_PAGE}\nvalidation={status}\nvideo_count={len(videos)}\nsize_bytes={size}\n")
    metadata = {"title": "Northwestern UCLA Multiview Action RGB - Base Paper", "id": DATASET_SLUG, "licenses": [{"name": "other"}]}
    (PUBLISH / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Kaggle CLI 2.x requires this plural filename.
    (PUBLISH / "datasets-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def create_dataset():
    try:
        print(run(["kaggle", "datasets", "list", "-m"], check=True)[:2000])
        print(run(["kaggle", "datasets", "create", "--help"], check=True))
        # If the exact owned slug exists, version it; otherwise make a private dataset (CLI default).
        listing = run(["kaggle", "datasets", "list", "-m"], check=True)
        if DATASET_SLUG in listing:
            print(run(["kaggle", "datasets", "version", "-p", str(PUBLISH), "-m", "Prepared official NW-UCLA RGB source"], check=True))
        else:
            print(run(["kaggle", "datasets", "create", "-p", str(PUBLISH), "-r", "tar"], check=True))
        return DATASET_SLUG
    except Exception as e:
        print(f"REMOTE_KAGGLE_DATASET_CREATE=UNAVAILABLE ({type(e).__name__}: {e})")
        print("PREPARED_AS_KERNEL_OUTPUT=YES")
        return "unavailable"


def main():
    kaggle_guard()
    for d in (DOWNLOADS, EXTRACTED, PUBLISH): d.mkdir(parents=True, exist_ok=True)
    s = session(); url = find_rgb_url(s); archive = download(s, url); extract(archive)
    files, videos, size, status = validate(); publish(files, videos, size, status)
    slug = create_dataset()
    if slug != "unavailable":
        archive.unlink(missing_ok=True); print("TEMPORARY_ARCHIVES_REMOVED=YES")
    print("REMOTE_PREP_STATUS=COMPLETE")
    print("DATASET_NAME=Northwestern-UCLA Multiview Action 3D")
    print("MODALITY=RGB")
    print(f"SOURCE={OFFICIAL_PAGE}")
    print(f"VIDEO_COUNT={len(videos)}")
    print(f"TOTAL_SIZE={human(size)}")
    print(f"PRIVATE_DATASET_SLUG={slug}")


if __name__ == "__main__": main()

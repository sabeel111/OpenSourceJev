import os
import sys
import time
import urllib.request
import hashlib
from pathlib import Path

URL = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf"
DEST = Path("models/qwen35-4b-q4km/Qwen3.5-4B-Q4_K_M.gguf")
EXPECTED_SIZE = 2740937888

def download():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temp_file = DEST.with_suffix(".gguf.part")
    
    current_size = temp_file.stat().st_size if temp_file.exists() else 0
    if DEST.exists() and DEST.stat().st_size == EXPECTED_SIZE:
        print(f"File already downloaded: {DEST} ({DEST.stat().st_size} bytes)")
        return DEST

    headers = {"User-Agent": "Mozilla/5.0"}
    if current_size > 0:
        headers["Range"] = f"bytes={current_size}-"
        print(f"Resuming download from byte {current_size} ({current_size / (1024**3):.2f} GB)...")
    else:
        print(f"Starting download of {DEST.name} ({EXPECTED_SIZE / (1024**3):.2f} GB)...")

    req = urllib.request.Request(URL, headers=headers)
    started = time.time()
    last_print = started
    downloaded = current_size

    mode = "ab" if current_size > 0 else "wb"
    with urllib.request.urlopen(req, timeout=60) as resp, open(temp_file, mode) as fh:
        while chunk := resp.read(1024 * 1024 * 4): # 4MB chunks
            fh.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print >= 5:
                pct = (downloaded / EXPECTED_SIZE) * 100
                speed_mb = (downloaded - current_size) / (now - started) / (1024 * 1024)
                eta_s = (EXPECTED_SIZE - downloaded) / ((downloaded - current_size) / (now - started) + 1e-6)
                print(f"[{pct:5.1f}%] {downloaded / (1024**3):.2f} / {EXPECTED_SIZE / (1024**3):.2f} GB | Speed: {speed_mb:.1f} MB/s | ETA: {eta_s:.0f}s")
                last_print = now

    print(f"Download complete: {downloaded} bytes.")
    temp_file.replace(DEST)
    return DEST

def verify(path: Path):
    print(f"Computing SHA-256 for {path}...")
    h = hashlib.sha256()
    size = path.stat().st_size
    with open(path, "rb") as f:
        read_bytes = 0
        last_print = time.time()
        while chunk := f.read(1024 * 1024 * 16):
            h.update(chunk)
            read_bytes += len(chunk)
            if time.time() - last_print >= 5:
                print(f"Hashing: {read_bytes / size * 100:.1f}%...")
                last_print = time.time()
    digest = h.hexdigest()
    print(f"SHA-256: {digest}")
    print(f"File Size: {size} bytes ({size / (1024**3):.2f} GB)")
    return digest

if __name__ == "__main__":
    dest = download()
    verify(dest)

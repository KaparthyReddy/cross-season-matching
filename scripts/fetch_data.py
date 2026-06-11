import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

SLICE_URL = "https://data.cmu.edu/data/published/CMU_Seasons/data/slice2.zip"
SLICE_ZIP = "slice2.zip"
EXTRACT_DIR = "slice2_raw"
SUMMER_OUT = "data/summer"
WINTER_OUT = "data/winter"
NUM_PAIRS = 5

def download_slice():
    if not os.path.exists(SLICE_ZIP):
        print("Downloading slice2.zip (~200MB)...")
        urllib.request.urlretrieve(SLICE_URL, SLICE_ZIP, reporthook=progress)
        print()
    else:
        print("slice2.zip already exists, skipping download.")

def progress(count, block_size, total_size):
    pct = min(int(count * block_size * 100 / total_size), 100)
    print(f"\r  {pct}%", end="", flush=True)

def extract_slice():
    if not os.path.exists(EXTRACT_DIR):
        print(f"Extracting to {EXTRACT_DIR}/...")
        with zipfile.ZipFile(SLICE_ZIP, "r") as z:
            z.extractall(EXTRACT_DIR)
    else:
        print(f"{EXTRACT_DIR}/ already exists, skipping extraction.")

def collect_pairs():
    os.makedirs(SUMMER_OUT, exist_ok=True)
    os.makedirs(WINTER_OUT, exist_ok=True)

    base = Path(EXTRACT_DIR)
    query_dir = next(base.rglob("query"), None)
    db_dir = next(base.rglob("database"), None)

    if query_dir is None or db_dir is None:
        raise RuntimeError(f"Could not find query/ or database/ inside {EXTRACT_DIR}. Check zip structure.")

    query_imgs = sorted(query_dir.glob("*.jpg"))[:NUM_PAIRS]
    db_imgs = sorted(db_dir.glob("*.jpg"))[:NUM_PAIRS]

    if len(query_imgs) < NUM_PAIRS or len(db_imgs) < NUM_PAIRS:
        raise RuntimeError(f"Not enough images found. query={len(query_imgs)} db={len(db_imgs)}")

    for i, (q, d) in enumerate(zip(query_imgs, db_imgs)):
        name = f"{i+1:04d}.jpg"
        shutil.copy(q, os.path.join(SUMMER_OUT, name))
        shutil.copy(d, os.path.join(WINTER_OUT, name))
        print(f"  Pair {i+1}: {q.name} -> summer/{name} | {d.name} -> winter/{name}")

def cleanup():
    print("Cleaning up raw extract...")
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    os.remove(SLICE_ZIP)

if __name__ == "__main__":
    download_slice()
    extract_slice()
    collect_pairs()
    cleanup()
    print(f"\nDone. {NUM_PAIRS} pairs ready in data/summer/ and data/winter/")

# %%
"""
Wallet Transaction Fetcher
===========================================

A cell‑oriented version of **wallet_fetcher** designed to be run inside
Jupyter Lab/Notebook or VS Code.  Each `# %%` delimiter marks an executable
cell.

* Loads *wallets_classes.csv* and filters for class `2` (Elliptic licit).
* Skips wallets already downloaded to *wallets/elliptic_licit*.
* Uses a **rotating proxy** (placeholder `http://YOUR_ROTATING_PROXY`).  Set
  the real gateway via env‑var `ROTATING_PROXY_URL`.
* Handles `429`/`50x` with exponential back‑off.
* Persists failures to `failed_wallets.json` so you can rerun the last cell
  to retry.
"""
# %%
from __future__ import annotations

import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence

import requests

# %%
DATA_FILE = "wallets_classes.csv"  # input CSV with `address, class` columns
CLASS_FILTER = 2                   # keep only this class label

OUTPUT_DIR = Path("wallets/elliptic_licit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rotating proxy gateway – replace or export ROTATING_PROXY_URL
PROXY_URL = os.getenv("ROTATING_PROXY_URL", "http://YOUR_ROTATING_PROXY")

MAX_WORKERS = 5            # parallel threads
BATCH_SIZE = 50            # wallets per batch
DELAY_BETWEEN_BATCHES = 2  # seconds between batches
REQUEST_TIMEOUT = 30       # HTTP timeout (s)
MAX_RETRIES = 3            # per‑request retries
BACKOFF_FACTOR = 1.5       # exponential back‑off multiplier
# %%

# %%
# ----------------------------------------------------------------------
# Load & filter wallet list
# ----------------------------------------------------------------------

def load_wallets(csv_path: str, wallet_class: int) -> List[str]:
    """Return wallet addresses whose *class* equals *wallet_class*."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        return [row["address"] for row in reader if int(row["class"]) == wallet_class]


def already_downloaded(out_dir: Path) -> set[str]:
    pattern = re.compile(r"(.+)_transactions\\.json$")
    return {
        pattern.match(p.name).group(1)
        for p in out_dir.glob("*_transactions.json")
        if pattern.match(p.name)
    }

wallets_all = load_wallets(DATA_FILE, CLASS_FILTER)
wallets_done = already_downloaded(OUTPUT_DIR)
wallets_to_process = [w for w in wallets_all if w not in wallets_done]

print(f"{len(wallets_done)} / {len(wallets_all)} wallets already downloaded")
print(f"{len(wallets_to_process)} queued for processing")

# %%
# ----------------------------------------------------------------------
# Helper functions – HTTP with retry & save
# ----------------------------------------------------------------------

def _request_with_retry(session: requests.Session, url: str, retries: int = MAX_RETRIES):
    """GET *url* with exponential back‑off on 429/5xx/network errors."""
    attempt = 0
    while attempt <= retries:
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504):
                delay = BACKOFF_FACTOR ** attempt
                print(f"⚠️ HTTP {resp.status_code} – retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                print(f"❌ Unhandled status {resp.status_code} – skipping")
                return None
        except requests.RequestException as exc:
            delay = BACKOFF_FACTOR ** attempt
            print(f"⚠️ {type(exc).__name__}: {exc} – retrying in {delay:.1f}s")
            time.sleep(delay)
        attempt += 1
    return None


def fetch_and_save(wallet: str, out_dir: Path, session: requests.Session) -> bool:
    url = f"https://blockchain.info/rawaddr/{wallet}"
    resp = _request_with_retry(session, url)
    if resp and resp.status_code == 200:
        out_path = out_dir / f"{wallet}_transactions.json"
        out_path.write_text(json.dumps(resp.json(), indent=2))
        print(f"✅ Saved: {out_path.name}")
        return True
    print(f"❌ Failed: {wallet}")
    return False


def process_wallets(wallets: Sequence[str], out_dir: Path):
    failed = []
    proxy_dict = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None
    with requests.Session() as session:
        if proxy_dict:
            session.proxies.update(proxy_dict)

        for batch_no, batch_start in enumerate(range(0, len(wallets), BATCH_SIZE), start=1):
            batch = wallets[batch_start : batch_start + BATCH_SIZE]
            print(f"\n=== Batch {batch_no} – {len(batch)} wallets ===")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(fetch_and_save, w, out_dir, session): w for w in batch}
                for fut in as_completed(futures):
                    if not fut.result():
                        failed.append(futures[fut])

            if batch_start + BATCH_SIZE < len(wallets):
                print(f"⏱️ Sleeping {DELAY_BETWEEN_BATCHES}s before next batch…")
                time.sleep(DELAY_BETWEEN_BATCHES)
    return failed

# %%
# ----------------------------------------------------------------------
# Execute download – run this cell
# ----------------------------------------------------------------------
failed_wallets = process_wallets(wallets_to_process, OUTPUT_DIR)

if failed_wallets:
    failed_path = OUTPUT_DIR / "failed_wallets.json"
    failed_path.write_text(json.dumps(sorted(failed_wallets), indent=2))
    print(f"\nℹ️  Wrote failed wallets list to {failed_path}")

print(f"\n✔️ Completed. Success: {len(wallets_to_process) - len(failed_wallets)}, Failed: {len(failed_wallets)}")

# %%
# ----------------------------------------------------------------------
# Retry block – rerun to process failed_wallets.json
# ----------------------------------------------------------------------
retry_path = OUTPUT_DIR / "failed_wallets.json"
if retry_path.exists():
    wallets_retry = json.loads(retry_path.read_text())
    print(f"Retrying {len(wallets_retry)} wallets…")
    failed_wallets = process_wallets(wallets_retry, OUTPUT_DIR)
    if failed_wallets:
        retry_path.write_text(json.dumps(sorted(failed_wallets), indent=2))
        print(f"Updated failed list – {len(failed_wallets)} wallets remain")
    else:
        retry_path.unlink()
        print("All retries succeeded – failed_wallets.json removed")
else:
    print("No failed_wallets.json found – nothing to retry.")




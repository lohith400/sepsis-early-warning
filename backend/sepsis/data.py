"""PhysioNet / CinC 2019 data access (Training Set A)."""
import glob
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/"


def download_set_a(data_dir: str, limit: int | None = None, workers: int = 16, log=print) -> int:
    """Download the .psv patient files in parallel (skips files that already exist)."""
    os.makedirs(data_dir, exist_ok=True)
    html = requests.get(BASE_URL, timeout=60).text
    names = sorted(set(re.findall(r"(p\d{6}\.psv)", html)))
    if limit:
        names = names[:limit]
    log(f"Files listed: {len(names)}")
    sess = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2)
    sess.mount("https://", adapter)

    def grab(n):
        path = os.path.join(data_dir, n)
        if os.path.exists(path):
            return
        for _ in range(4):
            try:
                r = sess.get(BASE_URL + n, timeout=60)
                if r.status_code == 200:
                    with open(path, "wb") as fh:
                        fh.write(r.content)
                    return
            except requests.RequestException:
                time.sleep(1)

    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(grab, names))
    log(f"Downloaded in {time.time() - t:.0f}s")
    return len(glob.glob(os.path.join(data_dir, "*.psv")))


def load_patients(data_dir: str, limit: int | None = None) -> pd.DataFrame:
    """Read all .psv files into one DataFrame with a ``pid`` column (rows stay in time order)."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.psv")))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .psv files in {data_dir}. Run with --download first.")
    frames = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, sep="|")
        d["pid"] = i
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

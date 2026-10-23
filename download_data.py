"""Download the 40 CWRU 12k drive-end files into data/cwru/ (parallel, resumable)."""
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cwru")
os.makedirs(DATA_DIR, exist_ok=True)

IDS = (
    [97, 98, 99, 100]                                        # Normal
    + [105, 106, 107, 108, 169, 170, 171, 172, 209, 210, 211, 212]  # Inner race
    + [118, 119, 120, 121, 185, 186, 187, 188, 222, 223, 224, 225]  # Ball
    + [130, 131, 132, 133, 197, 198, 199, 200, 234, 235, 236, 237]  # Outer race
)

URLS = [
    "https://engineering.case.edu/sites/default/files/{id}.mat",
    "https://csegroups.case.edu/sites/default/files/bearingdatacenter/files/Datafiles/{id}.mat",
]


def fetch(fid):
    path = os.path.join(DATA_DIR, f"{fid}.mat")
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return fid, True
    for u in URLS:
        try:
            urllib.request.urlretrieve(u.format(id=fid), path)
            if os.path.getsize(path) > 10_000:
                return fid, True
        except Exception:
            pass
    return fid, False


if __name__ == "__main__":
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(fetch, IDS))
    failed = [f for f, ok in results if not ok]
    print(f"Downloaded {len(IDS) - len(failed)}/{len(IDS)} files into {DATA_DIR}")
    if failed:
        print("Failed:", failed)
        print("Get them manually: https://engineering.case.edu/bearingdatacenter/download-data-file")

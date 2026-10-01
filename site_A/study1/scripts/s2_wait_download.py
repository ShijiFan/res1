"""Wait for the A0g0_* RTC jobs and download/extract them into ../products.

Records SHA-256 of every downloaded zip in ../products/download_manifest.json.
"""
import hashlib
import json
import zipfile
from pathlib import Path

import hyp3_sdk

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "products"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rec = json.loads(sorted((ROOT / "jobs").glob("submitted_*.json"))[-1].read_text(encoding="utf-8"))
    ids = [j["job_id"] for j in rec["jobs"]]
    hyp3 = hyp3_sdk.HyP3()
    batch = hyp3_sdk.Batch([hyp3.get_job_by_id(i) for i in ids])
    batch = hyp3.watch(batch, timeout=6 * 3600, interval=60)
    failed = [j for j in batch if not j.succeeded()]
    if failed:
        raise SystemExit(f"STOP: {len(failed)} jobs did not succeed: {[j.name for j in failed]}")
    PROD.mkdir(parents=True, exist_ok=True)
    manifest = []
    for job in batch:
        for z in job.download_files(PROD):
            z = Path(z)
            manifest.append({"job_name": job.name, "job_id": job.job_id, "file": z.name, "sha256": sha256(z)})
            with zipfile.ZipFile(z) as zf:
                zf.extractall(PROD)
    (PROD / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"downloaded {len(manifest)} files -> {PROD}")


if __name__ == "__main__":
    main()

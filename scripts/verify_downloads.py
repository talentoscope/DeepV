"""Verify downloaded dataset files under data/raw and write a report."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.downloaders import download_dataset as mod

root = Path("data") / "raw"
root.mkdir(parents=True, exist_ok=True)
report = []

for name, info in mod.DATASETS.items():
    ds_dir = root / name
    exists = ds_dir.exists()
    files = []
    total_size = 0
    nonempty_files = 0
    metadata_present = False
    if exists:
        for p in sorted(ds_dir.rglob("*")):
            if p.is_file():
                rel = p.relative_to(ds_dir)
                size = p.stat().st_size
                files.append((str(rel), size))
                total_size += size
                if size > 0:
                    nonempty_files += 1
                if rel.name == "metadata.json":
                    metadata_present = True
    report.append(
        {
            "dataset": name,
            "exists": exists,
            "num_files": len(files),
            "nonempty_files": nonempty_files,
            "total_size_bytes": total_size,
            "metadata_present": metadata_present,
            "note": info.get("note", ""),
        }
    )

# print a concise summary
ok = [r for r in report if r["exists"] and r["nonempty_files"] > 0]
missing = [r for r in report if not r["exists"]]
empty = [r for r in report if r["exists"] and r["nonempty_files"] == 0]

print(f"Datasets present with files: {len(ok)}")
print(f"Datasets missing: {len(missing)}")
print(f"Datasets present but empty: {len(empty)}")

for r in missing[:10]:
    print("MISSING:", r["dataset"])
for r in empty[:10]:
    print("EMPTY:", r["dataset"])

# write full JSON report
with open("verify_downloads_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("Wrote verify_downloads_report.json")

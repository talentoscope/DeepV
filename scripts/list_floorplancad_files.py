from pathlib import Path

p = Path("data/raw/floorplancad")
if not p.exists():
    print("MISSING")
    raise SystemExit(1)
files = list(p.rglob("*"))
files = [f for f in files if f.is_file()]
print("TOTAL_FILES", len(files))
# show first 50
for f in files[:50]:
    print(f)

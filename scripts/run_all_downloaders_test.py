"""Run dry-run tests for all dataset downloaders and report results."""

import traceback
from pathlib import Path

from dataset.downloaders import download
from dataset.downloaders import download_dataset as mod

results = {}
for name, info in mod.DATASETS.items():
    fn_name = f"download_{name}"
    if hasattr(mod, fn_name) or info.get("downloader") is not None:
        try:
            print("Testing", name)
            # call adapter in test mode
            try:
                download(name, output_dir=Path("./data"), test=True)
            except TypeError:
                # fallback to direct call
                fn = getattr(mod, fn_name)
                # build minimal kwargs
                import inspect

                sig = inspect.signature(fn)
                kwargs = {}
                if "output_dir" in sig.parameters:
                    kwargs["output_dir"] = Path("./data")
                if "test_mode" in sig.parameters:
                    kwargs["test_mode"] = True
                if "verify" in sig.parameters:
                    kwargs["verify"] = False
                fn(**kwargs)
            results[name] = ("ok", None)
        except Exception:
            tb = traceback.format_exc()
            results[name] = ("error", tb)
    else:
        results[name] = ("manual", None)

print("\nSummary:")
ok = [k for k, v in results.items() if v[0] == "ok"]
err = [k for k, v in results.items() if v[0] == "error"]
man = [k for k, v in results.items() if v[0] == "manual"]
print("OK:", len(ok), ok)
print("ERROR:", len(err), err)
print("MANUAL:", len(man), man)
for k in err[:10]:
    print("\n---", k, "traceback ---")
    print(results[k][1][:2000])

# Save full report
with open("downloaders_test_report.txt", "w", encoding="utf-8") as f:
    for k, v in results.items():
        f.write(f"{k}: {v[0]}\n")
        if v[1]:
            f.write(v[1])
            f.write("\n" + "-" * 80 + "\n")
print("Wrote downloaders_test_report.txt")

import importlib
import inspect

dd = importlib.import_module("dataset_downloaders.download_dataset")
funcs = [name for name, obj in inspect.getmembers(dd, inspect.isfunction) if name.startswith("download_")]
print("found", len(funcs), "downloaders")
for f in funcs:
    print("-", f)

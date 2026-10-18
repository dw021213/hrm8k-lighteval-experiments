"""E1: load HRM8K from the Hub exactly like Lighteval would, record schema facts."""
import json, re, collections
from datasets import load_dataset, get_dataset_config_names
REPO = "HAERAE-HUB/HRM8K"
out = {}
configs = get_dataset_config_names(REPO)
print("configs:", configs)
for cfg in configs:
    ds = load_dataset(REPO, cfg)
    info = {"splits": {k: len(v) for k, v in ds.items()}}
    test = ds["test"]
    info["features"] = {k: str(v) for k, v in test.features.items()}
    ans = test["answer"]
    info["answer_python_types"] = dict(collections.Counter(type(a).__name__ for a in ans))
    out[cfg] = info
    print(cfg, json.dumps(info, ensure_ascii=False))
    if cfg == "OMNI_MATH":
        big = max(ans, key=lambda a: float(a))
        print("  OMNI largest answer as loaded:", repr(big), "type", type(big).__name__,
              "| int(float(.)) =", int(float(big)))
json.dump(out, open("results/e1_schema.json", "w"), ensure_ascii=False, indent=1)

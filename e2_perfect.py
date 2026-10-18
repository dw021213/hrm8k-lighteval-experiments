"""E2: if a model writes the exact gold inside \\boxed{}, how often does each scorer accept it?"""
import json, sys, collections
from multiprocessing import Pool
from datasets import load_dataset

def work(args):
    import scorers
    cfg, idx, raw = args
    pred = scorers.perfect_prediction(raw)
    res = {}
    for name, f in scorers.SCORERS.items():
        try:
            res[name] = f(raw, pred)
        except Exception as e:          # record crashes as failures, keep the type
            res[name] = 0
            res[name + "__err"] = type(e).__name__
    return cfg, idx, raw if not isinstance(raw, float) else raw, res

if __name__ == "__main__":
    jobs = []
    for cfg in ["GSM8K", "MATH", "OMNI_MATH", "MMMLU", "KSM"]:
        ds = load_dataset("HAERAE-HUB/HRM8K", cfg, split="test")
        jobs += [(cfg, i, a) for i, a in enumerate(ds["answer"])]
    with Pool(8) as p:
        rows = p.map(work, jobs, chunksize=40)
    summary = collections.defaultdict(lambda: collections.Counter())
    fails = collections.defaultdict(list)
    for cfg, idx, raw, res in rows:
        summary[cfg]["n"] += 1
        for k, v in res.items():
            if k.endswith("__err"):
                summary[cfg][k + ":" + v] += 1
                continue
            summary[cfg][k] += v
            if not v:
                fails[(cfg, k)].append((idx, str(raw)))
    names = ["lmeval_hrm8k", "lighteval_expr_gold", "lighteval_latex_gold", "lighteval_latex_gold_$", "lighteval_both_gold_$", "proposed_rule"]
    print("subset".ljust(10), "n".rjust(5), *[n.rjust(24) for n in names])
    tot = collections.Counter()
    for cfg in ["GSM8K", "MATH", "OMNI_MATH", "MMMLU", "KSM"]:
        s = summary[cfg]; tot["n"] += s["n"]
        cells = []
        for n in names:
            tot[n] += s[n]
            cells.append(f"{s[n]}/{s['n']} ({100*s[n]/s['n']:.1f}%)".rjust(24))
        print(cfg.ljust(10), str(s["n"]).rjust(5), *cells)
    print("ALL".ljust(10), str(tot["n"]).rjust(5), *[f"{tot[n]}/{tot['n']} ({100*tot[n]/tot['n']:.1f}%)".rjust(24) for n in names])
    errs = {cfg: {k: v for k, v in summary[cfg].items() if "__err" in k} for cfg in summary}
    print("errors:", json.dumps(errs))
    json.dump({f"{c}|{k}": v[:40] for (c, k), v in fails.items()}, open("results/e2_fail_examples.json", "w"), ensure_ascii=False, indent=0)
    json.dump({c: dict(s) for c, s in summary.items()}, open("results/e2_summary.json", "w"), ensure_ascii=False, indent=1)

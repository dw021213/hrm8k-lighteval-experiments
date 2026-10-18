# HRM8K scorer experiments for Lighteval

Experiments behind the proposal to add [HRM8K](https://huggingface.co/datasets/HAERAE-HUB/HRM8K) (Korean math reasoning) to [Lighteval](https://github.com/huggingface/lighteval).
This is not the task implementation. It checks how candidate scorers grade HRM8K answers and lists data issues found along the way.
Korean notes: [README.ko.md](README.ko.md).

Pinned versions:
- dataset `HAERAE-HUB/HRM8K` @ `c360cabf8d733a82455565358b3dc965aab9ba8d`
- lighteval @ `c2af9e5f2fde494539f84c8ea9d09f18ac9e9c14`
- lm-evaluation-harness `hrm8k` scorer @ `314f717677d4aa0c706fc1c6bbd3c801d321384a` (copied to `third_party/`, MIT)

## Setup

```bash
pip install "lighteval @ git+https://github.com/huggingface/lighteval@c2af9e5f2fde494539f84c8ea9d09f18ac9e9c14" "huggingface_hub<2"
```

## Scripts

| Script | What it checks | Output |
|---|---|---|
| `e1_load.py` | Configs, splits, columns and `answer` dtype of each subset | `results/e1_load.txt` |
| `e2_perfect.py` | Writes every gold inside `\boxed{}` as a synthetic "perfect" answer and counts how often each scorer accepts it (all 8,011 items) | `results/e2_perfect.txt` |
| `e3_variants.py` | The same gold written differently (`-8 \pi`, `0.5` vs `\frac{1}{2}`, `2,880,000`, Korean units, ...) across scorers; also shows that `Language.KOREAN` crashes the extractive metric | `results/e3_variants.txt` |
| `e5_known_issues.py` | Recomputes the KSM rows reported as wrong (numeric integration, brute-force counting, Markov chain) | `results/e5_known_issues.txt` |
| `e6_korean_literals.py` | `openai_mmlu_kor_*` and the extractive metric before and after filling the Korean translation literals in memory | stdout |
| `hrm8k_proto.py` | Minimal custom task (lm-eval Korean prompt, gold parsed as LaTeX or expression) run end to end with a 135M model | `results/proto_run.txt` |
| `hrm8k_mc_proto.py` | MMMLU subset scored by option log-likelihood, for comparison only | `results/mc_proto.txt` |

```bash
python e2_perfect.py
lighteval accelerate "model_name=HuggingFaceTB/SmolLM2-135M-Instruct,batch_size=1" "hrm8k_proto|0" --custom-tasks hrm8k_proto.py --max-samples 2
```

## Scorers compared (`scorers.py`)

- `lmeval_hrm8k`: lm-evaluation-harness `hrm8k` `process_results` (last `\boxed{}`, MATH normalization, string match)
- `lighteval_expr_gold`: lighteval `Metrics.expr_gold_metric` (the one used by the `gsm8k` task)
- `proposed_rule`: lighteval `MultilingualExtractiveMatchMetric` with the gold written as an integer when possible, wrapped in `$…$` and parsed as expression or LaTeX; prediction parsed with `\boxed{}` first; `precision=5`

## Results (2026-10-16)

Acceptance of synthetic perfect answers (`e2_perfect.py`):

| Scorer | GSM8K | MATH | OMNI_MATH | MMMLU | KSM | All |
|---|---:|---:|---:|---:|---:|---:|
| lm-eval `hrm8k` | 99.8% | 100% | 100% | 100% | 83.6% | 97.1% |
| lighteval `expr_gold_metric` | 100% | 100% | 100% | 100% | 97.1% | 99.5% |
| proposed rule | 100% | 100% | 100% | 100% | 99.9% | 100.0% |

- lm-eval runs its answer extractor on the gold too, so `-10` becomes `10` and `\frac{1}{2}` becomes `2`; 236 items (GSM8K 2, KSM 234) can never be matched.
- The proposed rule misses one item, KSM row 201, whose gold `\frac[13}{20}` is malformed.
- This test only checks that a scorer can read each gold. It cannot detect golds that are readable but wrong; see `e5_known_issues.py`.

KSM rows (0-based, revision `c360cab`) with wrong or suspicious golds:

| Row | Stored answer | Finding |
|---|---|---|
| 3 | `\frac{x-3}{4}` | definite integral; value is (π−3)/4 |
| 109 | `\sqrt{1+r^2}` | problem uses τ; value is √(1+τ²) |
| 201 | `\frac[13}{20}` | malformed LaTeX; value is 13/20 |
| 1427 | `\frac{6}{(1-x)^4}` | asks for a probability; value is 40/81 |
| 1262 | `3\times4^p` | p is not in the problem; same gold as row 1258; brute force suggests 3n², i.e. 12,241,200 for n = 2020 |
| 1254 | `C_n` | symbol not defined in the problem |
| 323 | `answer` 1, `original_answer` 4 | `answer` is correct; `original_answer` is wrong |

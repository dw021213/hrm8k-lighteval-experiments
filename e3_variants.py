"""E3: same gold, differently written predictions -> which scorer accepts them?"""
import scorers
from scorers import SCORERS
NAMES = ["lmeval_hrm8k", "lighteval_expr_gold", "lighteval_both_gold_$"]
def wrap(p):  # put the boxed answer inside a short Korean CoT ending, like a real model output
    return p if not p.startswith("\\boxed") else f"계산하면 다음과 같다.\n따라서 최종 답변은 ${p}$ 입니다."
CASES = [
    ("GSM8K", 18.0, ["\\boxed{18}", "\\boxed{18.0}", "\\boxed{18달러}", "\\boxed{\\$18}", "정답: 18", "따라서 답은 18입니다.",
                     "답은 18달러이다. 검산: 9×2=18"]),
    ("GSM8K", 2880000.0, ["\\boxed{2880000}", "\\boxed{2,880,000}", "\\boxed{\\$2,880,000}", "\\boxed{288만}"]),
    ("GSM8K", -10.0, ["\\boxed{-10}", "\\boxed{−10}"]),
    ("GSM8K", 5.0, ["먼저 \\boxed{3}을 구하고, 최종 답은 \\boxed{5}", "최종 답은 \\boxed{5}이고, 참고로 \\boxed{3}"]),
    ("KSM", "-8\\pi", ["\\boxed{-8\\pi}", "\\boxed{-8 \\pi}", "\\boxed{-8π}", "\\boxed{-25.13}", "\\boxed{-25.1327}", "\\boxed{-\\pi \\cdot 8}"]),
    ("KSM", "0.5", ["\\boxed{\\frac{1}{2}}", "\\boxed{1/2}", "\\boxed{0.5}", "\\boxed{0.50}"]),
    ("KSM", "\\frac{1}{2}", ["\\boxed{0.5}", "\\boxed{1/2}", "\\boxed{\\dfrac{1}{2}}", "\\boxed{\\frac12}"]),
    ("KSM", "-1", ["\\boxed{-1}"]),
    ("MMMLU", 4, ["\\boxed{4}", "\\boxed{④}", "\\boxed{(4)}", "\\boxed{4번}", "\\boxed{D}", "정답은 4번입니다."]),
    ("OMNI_MATH", 66071772829247409, ["\\boxed{66071772829247409}", "\\boxed{66071772829247408}"]),
]
rows = []
for subset, gold, preds in CASES:
    for p in preds:
        text = wrap(p)
        res = []
        for n in NAMES:
            try: res.append("O" if SCORERS[n](gold, text) else "X")
            except Exception as e: res.append("ERR:" + type(e).__name__)
        rows.append((subset, str(gold), p, *res))
w = [9, 18, 34, 14, 22, 22]
hdr = ("subset", "gold", "prediction (inside the answer)", "lm-eval", "lighteval expr_gold", "lighteval both_gold_$")
print(" | ".join(h.ljust(x) for h, x in zip(hdr, w)))
for r in rows: print(" | ".join(str(c).ljust(x) for c, x in zip(r, w)))

print("\n=== E4: Language.KOREAN in MultilingualExtractiveMatchMetric")
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.utils.extractive_match_utils import ExprExtractionConfig, LatexExtractionConfig
from lighteval.utils.language import Language
for tgt_name, tgt in [("Expr only", (ExprExtractionConfig(),)),
                      ("Latex(boxed first)", (LatexExtractionConfig(boxed_match_priority=0),))]:
    m = MultilingualExtractiveMatchMetric(language=Language.KOREAN, gold_extraction_target=(ExprExtractionConfig(),),
                                          pred_extraction_target=tgt, precision=5)
    try:
        print(tgt_name, "->", scorers._lt(m, "18", wrap("\\boxed{18}")))
    except Exception as e:
        print(tgt_name, "->", type(e).__name__, str(e).strip().splitlines()[0][:120])

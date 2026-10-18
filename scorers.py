"""Shared scorers for the HRM8K grading experiments (not part of any repo)."""
import importlib.util, logging, os
logging.disable(logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "lmeval_hrm8k_utils", os.path.join(HERE, "third_party", "lmeval_hrm8k_utils.py"))
lmeval = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lmeval)

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.utils.extractive_match_utils import ExprExtractionConfig, LatexExtractionConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc

_expr_gold = Metrics.expr_gold_metric.value.sample_level_fn          # what lighteval's gsm8k task uses
_latex_gold = MultilingualExtractiveMatchMetric(                     # gold parsed as LaTeX
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    precision=5)
_both_gold = MultilingualExtractiveMatchMetric(                      # gold parsed as expr OR LaTeX
    gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    precision=5)

def _lt(metric, gold: str, pred: str) -> int:
    doc = Doc(query="", choices=[gold], gold_index=0)
    return int(metric.compute(doc=doc, model_response=ModelResponse(text=[pred])) == 1.0)

SCORERS = {
    # name: f(raw_answer_as_loaded_from_hub, prediction_text) -> 0/1
    "lmeval_hrm8k":            lambda raw, pred: lmeval.process_results({"answer": raw}, [pred])["exact_match"],
    "lighteval_expr_gold":     lambda raw, pred: _lt(_expr_gold, str(raw), pred),
    "lighteval_latex_gold":    lambda raw, pred: _lt(_latex_gold, str(raw), pred),
    "lighteval_latex_gold_$":  lambda raw, pred: _lt(_latex_gold, f"${raw}$", pred),
    "lighteval_both_gold_$":   lambda raw, pred: _lt(_both_gold, f"${raw}$", pred),
}

# Exactly the rule proposed in analysis.md 5-4: integer-valued floats written as ints, gold wrapped in $...$
import sys as _sys
_sys.path.insert(0, HERE)
from hrm8k_proto import _gold as _proposed_gold
SCORERS["proposed_rule"] = lambda raw, pred: _lt(_both_gold, _proposed_gold(raw), pred)

def canonical(raw) -> str:
    """How a model would naturally write the gold inside \\boxed{} (12.0 -> 12)."""
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    return str(raw)

def perfect_prediction(raw) -> str:
    return f"풀이를 정리하면 다음과 같습니다.\n따라서 최종 답변은 $\\boxed{{{canonical(raw)}}}$ 입니다."

"""E6: what breaks because Lighteval's Korean translation literals are incomplete, and what an in-memory fill fixes.

Nothing in the lighteval repo is modified: the Korean entry is replaced in memory only.
"""
import logging

from datasets import load_dataset

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.utils.extractive_match_utils import ExprExtractionConfig, LatexExtractionConfig
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.multilingual.tasks import openai_mmlu as om
from lighteval.tasks.requests import Doc
from lighteval.tasks.templates.utils import translation_literals as tl
from lighteval.utils.language import Language


logging.disable(logging.WARNING)

row = [r for r in load_dataset("openai/MMMLU", "KO_KR", split="test") if r["Subject"] == "high_school_mathematics"][0]
NAME = "openai_mmlu_kor_mcf:high_school_mathematics"
cfg = {c.name: c for c in om.TASKS_TABLE}[NAME]
PREDS = [
    "따라서 최종 답변은 $\\boxed{18}$ 입니다.",
    "계산하면 9×2=18이다. 답변: 18 (검산: 16-3-4=9)",
    "계산하면 9×2=18이다. 정답: 18 (검산: 16-3-4=9)",
]


def run(label):
    print(f"\n######## {label}")
    try:
        d = cfg.prompt_function(row, NAME)
        print(f"[{NAME}] query:\n{d.query}\n  choices={d.choices} gold_index={d.gold_index}")
    except Exception as e:
        print(f"[{NAME}] {type(e).__name__}: {str(e).strip().splitlines()[0]}")
    metric = MultilingualExtractiveMatchMetric(
        language=Language.KOREAN,
        gold_extraction_target=(ExprExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        precision=5,
    )
    for pred in PREDS:
        try:
            d = Doc(query="", choices=["18"], gold_index=0)
            score = metric.compute(doc=d, model_response=ModelResponse(text=[pred]))
            print(f"[metric, Language.KOREAN] score={score} extracted={d.specific['extracted_predictions']} <- {pred!r}")
        except Exception as e:
            print(f"[metric, Language.KOREAN] {type(e).__name__}: {str(e).strip().splitlines()[0]}")


run("BEFORE: lighteval c2af9e5 as is")
# Words proposed in huggingface/lighteval#1179, plus and_word, which #1179 does not add but lazy_latex_regex needs.
tl.TRANSLATION_LITERALS[Language.KOREAN] = tl.TranslationLiterals(
    language=Language.KOREAN, question_word="질문", answer="답변", confirmation_word="맞죠", yes="예", no="아니오",
    also="또한", cause_word="왜냐하면", effect_word="그래서", or_word="또는", and_word="그리고",
    true="참", false="거짓", neither="둘 다 아님",
)
run("AFTER: Korean literals filled in memory")

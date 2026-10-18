"""Throwaway feasibility check for HRM8K in Lighteval (NOT the team implementation).

Prompt = lm-eval-harness `hrm8k` Korean prompt; gold wrapped in $...$ and parsed as LaTeX or expression.
"""
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.utils.extractive_match_utils import ExprExtractionConfig, LatexExtractionConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.metrics.utils.metric_utils import SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
import numpy as np

INSTRUCTION = "주어진 문제를 풀어보세요.\n문제를 푼 후, 최종 답변을 다음과 같은 형식으로 작성하세요: $\\boxed{N}$.\n\n"
INSTRUCTION_MMMLU = ("주어진 문제를 풀어보세요.\n문제를 푼 후, 주어진 선택지 (1, 2, 3, 4) 중 최종 선택지를 "
                     "다음 형식으로 작성하세요: $\\boxed{N}$.\n\n")

def _gold(answer) -> str:
    if isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    return f"${answer}$"

def hrm8k_prompt(line, task_name: str = None):
    instruction = INSTRUCTION_MMMLU if task_name.endswith("mmmlu") else INSTRUCTION
    return Doc(task_name=task_name,
               query=f"{instruction}문제: {line['question'].strip()}\n답변:",
               choices=[_gold(line["answer"])],
               gold_index=0)

hrm8k_match = SampleLevelMetric(
    metric_name="extractive_match",
    sample_level_fn=MultilingualExtractiveMatchMetric(
        gold_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
        precision=5),
    category=SamplingMethod.GENERATIVE, corpus_level_fn=np.mean, higher_is_better=True)

TASKS_TABLE = [
    LightevalTaskConfig(
        name=f"hrm8k_proto:{subset.lower()}", prompt_function=hrm8k_prompt,
        hf_repo="HAERAE-HUB/HRM8K", hf_subset=subset,
        hf_avail_splits=["test"], evaluation_splits=["test"], few_shots_split=None, few_shots_select=None,
        generation_size=256, stop_sequence=[], metrics=[hrm8k_match], version=0)
    for subset in ["GSM8K", "MATH", "OMNI_MATH", "MMMLU", "KSM"]
]

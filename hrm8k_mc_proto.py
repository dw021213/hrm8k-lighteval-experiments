"""Throwaway check: HRM8K MMMLU scored by option log-likelihood (like openai_mmlu / lm-eval kmmlu), not the team implementation."""
from lighteval.metrics.dynamic_metrics import LogLikelihoodAccMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


def hrm8k_mmmlu_mc_prompt(line, task_name: str = None):
    # `question` already lists the options as "1. ...\n2. ...\n3. ...\n4. ..."
    return Doc(
        task_name=task_name,
        query=f"{line['question'].strip()}\n정답:",
        choices=[" 1", " 2", " 3", " 4"],
        gold_index=int(line["answer"]) - 1,
    )


TASKS_TABLE = [
    LightevalTaskConfig(
        name="hrm8k_mc_proto:mmmlu",
        prompt_function=hrm8k_mmmlu_mc_prompt,
        hf_repo="HAERAE-HUB/HRM8K",
        hf_subset="MMMLU",
        hf_revision="c360cabf8d733a82455565358b3dc965aab9ba8d",
        hf_avail_splits=["test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select=None,
        metrics=[LogLikelihoodAccMetric(normalization=None)],
        version=0,
    )
]

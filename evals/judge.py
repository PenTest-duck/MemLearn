"""
Judge module for evaluating MemLearn agent responses.

This module provides LLM-based evaluation of agent responses using prompts
from the Hindsight paper. Categories map to different evaluation criteria.

Categories:
1. Single-hop - Basic fact retrieval
2. Multi-hop - Multi-step reasoning
3. Temporal-knowledge - Time-based reasoning (with off-by-one tolerance)
4. Open-domain - Preference/personalization
5. Adversarial - Abstention (correctly identifying unanswerable questions)
"""

import os
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openai:gpt-5-mini")


class QuestionCategory(IntEnum):
    """LoCoMo question categories."""

    SINGLE_HOP = 1
    MULTI_HOP = 2
    TEMPORAL_KNOWLEDGE = 3
    OPEN_DOMAIN = 4
    ADVERSARIAL = 5


JUDGE_PROMPT_SINGLE_MULTI_SESSION = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the
response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct
answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If
the response only contains a subset of the information required by the answer, answer no.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Is the model response correct?
You may provide reasoning, but you MUST end your response with your final answer in the format:
\\boxed{{yes}} or \\boxed{{no}}
"""

JUDGE_PROMPT_TEMPORAL_REASONING = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the
response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct
answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If
the response only contains a subset of the information required by the answer, answer no. In addition,
do not penalize off-by-one errors for the number of days. If the question asks for the number of
days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the
answer is 18), the model's response is still correct.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Is the model response correct?
You may provide reasoning, but you MUST end your response with your final answer in the format:
\\boxed{{yes}} or \\boxed{{no}}
"""

JUDGE_PROMPT_KNOWLEDGE_UPDATE = """
I will give you a question, a correct answer, and a response from a model. Please answer yes if the
response contains the correct answer. Otherwise, answer no. If the response contains some previous
information along with an updated answer, the response should be considered as correct as long as the
updated answer is the required answer.

Question: {question}
Correct Answer: {answer}
Model Response: {response}

Is the model response correct?
You may provide reasoning, but you MUST end your response with your final answer in the format:
\\boxed{{yes}} or \\boxed{{no}}
"""

JUDGE_PROMPT_PREFERENCE = """
I will give you a question, a rubric for desired personalized response, and a response from a model.
Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does
not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes
the user's personal information correctly.

Question: {question}
Rubric: {answer}
Model Response: {response}

Is the model response correct?
You may provide reasoning, but you MUST end your response with your final answer in the format:
\\boxed{{yes}} or \\boxed{{no}}
"""

JUDGE_PROMPT_ABSTENTION = """
I will give you an unanswerable question, an explanation, and a response from a model. Please answer
yes if the model correctly identifies the question as unanswerable. The model could say that the
information is incomplete, or some other information is given but the asked information is not.

Question: {question}
Explanation: {answer}
Model Response: {response}

Does the model correctly identify the question as unanswerable?
You may provide reasoning, but you MUST end your response with your final answer in the format:
\\boxed{{yes}} or \\boxed{{no}}
"""


CATEGORY_PROMPTS = {
    QuestionCategory.SINGLE_HOP: JUDGE_PROMPT_SINGLE_MULTI_SESSION,
    QuestionCategory.MULTI_HOP: JUDGE_PROMPT_SINGLE_MULTI_SESSION,
    QuestionCategory.TEMPORAL_KNOWLEDGE: JUDGE_PROMPT_TEMPORAL_REASONING,
    QuestionCategory.OPEN_DOMAIN: JUDGE_PROMPT_PREFERENCE,
    QuestionCategory.ADVERSARIAL: JUDGE_PROMPT_ABSTENTION,
}


@dataclass
class JudgeResult:
    """Result from the judge evaluation."""

    question: str
    expected_answer: str
    model_response: str
    category: int
    is_correct: bool
    judge_reasoning: str
    raw_judge_response: str


def get_prompt_for_category(category: int) -> str:
    """Get the appropriate judge prompt for a question category."""
    try:
        cat = QuestionCategory(category)
        return CATEGORY_PROMPTS[cat]
    except ValueError:
        return JUDGE_PROMPT_SINGLE_MULTI_SESSION


def parse_judge_response(response: str) -> bool:
    """Parse the judge's response to extract yes/no verdict."""
    boxed_pattern = r"\\boxed\{(yes|no)\}"
    matches = re.findall(boxed_pattern, response.lower())

    if matches:
        return matches[-1] == "yes"

    response_lower = response.lower()
    if "\\boxed{yes}" in response_lower or "boxed{yes}" in response_lower:
        return True
    if "\\boxed{no}" in response_lower or "boxed{no}" in response_lower:
        return False

    lines = response.strip().split("\n")
    for line in reversed(lines):
        line_lower = line.lower().strip()
        if "yes" in line_lower and "no" not in line_lower:
            return True
        if "no" in line_lower and "yes" not in line_lower:
            return False

    return False


class Judge:
    """LLM-based judge for evaluating agent responses."""

    def __init__(self, model_name: str = JUDGE_MODEL):
        """Initialize the judge with a specific model."""
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazily initialize the model."""
        if self._model is None:
            self._model = init_chat_model(self.model_name, streaming=False)
        return self._model

    def evaluate(
        self,
        question: str,
        expected_answer: str | int | float,
        model_response: str,
        category: int,
    ) -> JudgeResult:
        """
        Evaluate a model response against the expected answer.

        Args:
            question: The question that was asked
            expected_answer: The ground truth answer
            model_response: The model's response
            category: Question category (1-5)

        Returns:
            JudgeResult with evaluation details
        """
        prompt_template = get_prompt_for_category(category)

        answer_str = str(expected_answer)

        prompt = prompt_template.format(
            question=question,
            answer=answer_str,
            response=model_response,
        )

        try:
            response = self.model.invoke(prompt)
            raw_response = response.content
            is_correct = parse_judge_response(raw_response)

            reasoning = raw_response
            if "\\boxed" in raw_response:
                reasoning = raw_response.split("\\boxed")[0].strip()

        except Exception as e:
            raw_response = f"Error: {str(e)}"
            is_correct = False
            reasoning = f"Judge failed with error: {str(e)}"

        return JudgeResult(
            question=question,
            expected_answer=answer_str,
            model_response=model_response,
            category=category,
            is_correct=is_correct,
            judge_reasoning=reasoning,
            raw_judge_response=raw_response,
        )

    def evaluate_batch(
        self,
        qa_pairs: list[dict[str, Any]],
        responses: list[str],
    ) -> list[JudgeResult]:
        """
        Evaluate a batch of question-answer pairs.

        Args:
            qa_pairs: List of dicts with 'question', 'answer', 'category'
            responses: List of model responses corresponding to each qa_pair

        Returns:
            List of JudgeResult objects
        """
        if len(qa_pairs) != len(responses):
            raise ValueError("qa_pairs and responses must have the same length")

        results = []
        for qa, response in zip(qa_pairs, responses):
            result = self.evaluate(
                question=qa["question"],
                expected_answer=qa["answer"],
                model_response=response,
                category=qa.get("category", 1),
            )
            results.append(result)

        return results


def calculate_metrics(results: list[JudgeResult]) -> dict[str, Any]:
    """
    Calculate evaluation metrics from judge results.

    Args:
        results: List of JudgeResult objects

    Returns:
        Dictionary with overall and per-category metrics
    """
    if not results:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "by_category": {}}

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)

    by_category: dict[int, dict[str, int]] = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = {"total": 0, "correct": 0}
        by_category[r.category]["total"] += 1
        if r.is_correct:
            by_category[r.category]["correct"] += 1

    category_metrics = {}
    category_names = {
        1: "single_hop",
        2: "multi_hop",
        3: "temporal",
        4: "open_domain",
        5: "adversarial",
    }

    for cat, counts in by_category.items():
        cat_name = category_names.get(cat, f"category_{cat}")
        category_metrics[cat_name] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "accuracy": (
                counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
            ),
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "by_category": category_metrics,
    }

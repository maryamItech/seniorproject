# """
# RAGAS-based evaluation helpers for faithfulness and answer relevancy.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any
# import os
# import re


# @dataclass
# class EvaluationResult:
#     faithfulness: float
#     answer_relevancy: float
#     used_ragas: bool
#     error: str | None = None


# def _tokenize_arabic(text: str) -> set[str]:
#     cleaned = re.sub(r"[^\u0600-\u06FF0-9A-Za-z\s]", " ", (text or ""))
#     tokens = [t for t in cleaned.split() if len(t) > 1]
#     return set(tokens)


# def _local_overlap_scores(question: str, answer: str, contexts: list[str]) -> EvaluationResult:
#     """
#     OpenAI-free fallback scoring for local/offline setups.
#     """
#     answer_tokens = _tokenize_arabic(answer)
#     question_tokens = _tokenize_arabic(question)
#     context_tokens = _tokenize_arabic(" ".join(contexts))
#     if not answer_tokens:
#         return EvaluationResult(faithfulness=0.0, answer_relevancy=0.0, used_ragas=False)
#     faith = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))
#     relevancy = len(answer_tokens & question_tokens) / max(1, len(question_tokens))
#     return EvaluationResult(
#         faithfulness=max(0.0, min(1.0, faith)),
#         answer_relevancy=max(0.0, min(1.0, relevancy)),
#         used_ragas=False,
#         error=None,
#     )


# def evaluate_with_ragas(
#     question: str,
#     answer: str,
#     contexts: list[str],
# ) -> EvaluationResult:
#     """
#     Evaluate a single QA turn using RAGAS metrics.
#     Uses native RAGAS when available; otherwise falls back to local scores.
#     """
#     try:
#         if not contexts:
#             return EvaluationResult(
#                 faithfulness=0.0,
#                 answer_relevancy=0.0,
#                 used_ragas=False,
#                 error="لا يمكن التقييم لعدم وجود مصادر مسترجعة",
#             )
#         use_openai = bool(os.getenv("OPENAI_API_KEY"))
#         if not use_openai:
#             return _local_overlap_scores(question=question, answer=answer, contexts=contexts)

#         from datasets import Dataset
#         from ragas import evaluate
#         from ragas.metrics import answer_relevancy, faithfulness

#         # Keep evaluation lightweight to reduce token usage/cost.
#         trimmed_contexts = [c[:700] for c in contexts[:2]]
#         if not trimmed_contexts:
#             trimmed_contexts = [""]
#         eval_payload = {
#             "question": [question[:1000]],
#             "contexts": [trimmed_contexts],
#             "answer": [answer[:2000]],
#             "ground_truth": [trimmed_contexts[0][:500]],
#         }
#         dataset = Dataset.from_dict(eval_payload)
#         result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy])
#         faith_score = float(result["faithfulness"])
#         rel_score = float(result["answer_relevancy"])
#         return EvaluationResult(
#             faithfulness=max(0.0, min(1.0, faith_score)),
#             answer_relevancy=max(0.0, min(1.0, rel_score)),
#             used_ragas=True,
#         )
#     except Exception as exc:
#         fallback = _local_overlap_scores(question=question, answer=answer, contexts=contexts)
#         fallback.error = f"RAGAS failed, local scoring used: {str(exc)}"
#         return fallback


# def append_eval_log(session_state: Any, result: EvaluationResult, question: str) -> None:
#     if "eval_logs" not in session_state:
#         session_state.eval_logs = []
#     session_state.eval_logs.append(
#         {
#             "question": question,
#             "faithfulness": result.faithfulness,
#             "answer_relevancy": result.answer_relevancy,
#             "used_ragas": result.used_ragas,
#             "error": result.error,
#         }
#     )
"""
RAGAS-based evaluation helpers for faithfulness and answer relevancy.
Optimized for Arabic legal terminology.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os
import re

@dataclass
class EvaluationResult:
    faithfulness: float
    answer_relevancy: float
    used_ragas: bool
    error: str | None = None

def _tokenize_arabic(text: str) -> set[str]:
    """Prepare Arabic text for evaluation by extracting keywords only."""
    if not text:
        return set()

# Clean text while preserving Arabic characters and numbers
    cleaned = re.sub(r"[^\u0600-\u06FF0-9]", " ", text)

    stop_words = {"من", "إلى", "على", "في", "عن", "مع", "هذا", "تم", "كان", "أو", "إن"}

    tokens = [t for t in cleaned.split() if len(t) > 1 and t not in stop_words]
    return set(tokens)

def _local_overlap_scores(question: str, answer: str, contexts: list[str]) -> EvaluationResult:
    """
    Fast local evaluation (offline) based on keyword overlap.
    Ideal for graduation projects to ensure stable and consistent performance.

    """
    answer_tokens = _tokenize_arabic(answer)
    question_tokens = _tokenize_arabic(question)
    context_tokens = _tokenize_arabic(" ".join(contexts))

    if not answer_tokens:
        return EvaluationResult(faithfulness=0.0, answer_relevancy=0.0, used_ragas=False)

    # Calculate Faithfulness: How many words in the answer are actually present in the retrieved legal articles?
    faith = len(answer_tokens & context_tokens) / max(1, len(answer_tokens))

    # Calculate Relevancy: How many words in the answer are related to the user's question?
    relevancy = len(answer_tokens & question_tokens) / max(1, len(question_tokens))
    return EvaluationResult(
        faithfulness=round(max(0.0, min(1.0, faith)), 2),
        answer_relevancy=round(max(0.0, min(1.0, relevancy)), 2),
        used_ragas=False,
        error=None,
    )

def evaluate_with_ragas(
    question: str,
    answer: str,
    contexts: list[str],
) -> EvaluationResult:
    """
    Local evaluation has been enabled as the default option to ensure speed and cost-effectiveness.
    """
    try:
        if not contexts:
            return EvaluationResult(
                faithfulness=0.0,
                answer_relevancy=0.0,
                used_ragas=False,
                error="لا يمكن التقييم لعدم وجود مصادر مسترجعة",
            )

        use_openai = bool(os.getenv("OPENAI_API_KEY"))
        if not use_openai:
            return _local_overlap_scores(question=question, answer=answer, contexts=contexts)

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        trimmed_contexts = [c[:700] for c in contexts[:2]]
        eval_payload = {
            "question": [question[:1000]],
            "contexts": [trimmed_contexts],
            "answer": [answer[:2000]],
        }
        dataset = Dataset.from_dict(eval_payload)
        result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy])

        return EvaluationResult(
            faithfulness=max(0.0, min(1.0, float(result["faithfulness"]))),
            answer_relevancy=max(0.0, min(1.0, float(result["answer_relevancy"]))),
            used_ragas=True,
        )

    except Exception as exc:
        fallback = _local_overlap_scores(question=question, answer=answer, contexts=contexts)
        fallback.error = f"Local scoring used (RAGAS skipped/failed)"
        return fallback

def append_eval_log(session_state: Any, result: EvaluationResult, question: str) -> None:
    if "eval_logs" not in session_state:
        session_state.eval_logs = []
    session_state.eval_logs.append(
        {
            "question": question,
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "used_ragas": result.used_ragas,
            "error": result.error,
        }
    )
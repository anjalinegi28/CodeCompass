"""
The eval-gate script. This is what GitHub Actions runs on every pull request.

Steps:
1. Load the fixed question set (app/eval/testset.json)
2. Run every question through the live RAG agent (app.rag.agent.ask)
3. Score the resulting answers with RAGAS: faithfulness, context precision,
   context recall
4. Log chunk size, embedding model, LLM provider, and the resulting scores
   to MLflow, so there is a full experiment history over time
5. Exit with status code 1 (failing the CI check, blocking the merge) if
   ANY score falls below app.config.settings.eval_threshold

Run locally with:  python -m app.eval.ragas_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from app.config import settings
from app.mlflow_logging.logger import log_eval_run
from app.rag.agent import ask

TESTSET_PATH = Path(__file__).parent / "testset.json"


def _load_testset() -> list[dict]:
    with open(TESTSET_PATH) as f:
        return json.load(f)


def _run_ragas(questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """
    Computes RAGAS scores. Imported lazily since ragas pulls in a fair
    amount of machinery that isn't needed for the rest of the app.

    Note: we do NOT use the langchain-google-genai bridge package here.
    That package's version requirements clash constantly with ragas and
    langchain-core (different langchain-core ranges, different underlying
    Google SDKs). Instead we call Gemini directly through our own, already
    working app/rag/providers.py, wrapped in a small adapter that satisfies
    ragas's BaseRagasLLM interface. This removes an entire class of
    version-conflict bugs.
    """
    import asyncio

    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms.base import BaseRagasLLM
    from ragas.metrics import context_precision, context_recall, faithfulness
    from langchain_core.outputs import Generation, LLMResult

    class _GeminiRagasLLM(BaseRagasLLM):
        """Adapts our existing app.rag.providers.complete(provider="gemini")
        call to the interface ragas expects from an evaluator LLM."""

        def _call_gemini(self, prompt_text: str) -> str:
            from app.rag.providers import complete

            result = complete(
                system_prompt="You are a careful, precise evaluation assistant.",
                user_prompt=prompt_text,
                provider="gemini",
            )
            return result.text

        def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
            text = self._call_gemini(prompt.to_string())
            return LLMResult(generations=[[Generation(text=text)] * n])

        async def agenerate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None) -> LLMResult:
            text = await asyncio.to_thread(self._call_gemini, prompt.to_string())
            return LLMResult(generations=[[Generation(text=text)] * n])

    evaluator_llm = _GeminiRagasLLM()

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=evaluator_llm,
    )
    return {k: float(v) for k, v in result.items()}


def run_eval() -> dict:
    testset = _load_testset()

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in testset:
        result = ask(item["question"])
        questions.append(item["question"])
        answers.append(result.text)
        contexts.append([s.text for s in result.sources])
        ground_truths.append(item["ground_truth"])

    scores = _run_ragas(questions, answers, contexts, ground_truths)

    log_eval_run(
        params={
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding_model": settings.embedding_model,
            "vector_store": settings.vector_store,
            "llm_provider": settings.llm_provider,
        },
        metrics=scores,
    )

    return scores


def main() -> int:
    scores = run_eval()

    print("\nRAGAS evaluation results")
    print("-" * 40)
    failed = []
    for metric, value in scores.items():
        status = "PASS" if value >= settings.eval_threshold else "FAIL"
        if status == "FAIL":
            failed.append(metric)
        print(f"{metric:<20} {value:.3f}   [{status}]  (threshold {settings.eval_threshold})")
    print("-" * 40)

    if failed:
        print(f"\nEval gate FAILED: {', '.join(failed)} below threshold. Blocking merge.")
        return 1

    print("\nEval gate PASSED. Safe to merge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
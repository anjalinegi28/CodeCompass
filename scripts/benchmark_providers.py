"""
Benchmarks OpenAI, Gemini, and Claude against the same eval question set:
records latency directly, and uses one provider as an "LLM-as-judge" to
score each answer's quality 1-5. Run after ingesting a project:

    python scripts/benchmark_providers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval.ragas_eval import TESTSET_PATH  # noqa: E402
from app.rag.agent import ask  # noqa: E402
from app.rag.providers import complete  # noqa: E402

JUDGE_PROVIDER = "claude"  # whichever provider you trust most to grade the others

JUDGE_SYSTEM_PROMPT = (
    "You are grading an AI assistant's answer to a question about a codebase. "
    "Score the answer's quality from 1 (bad) to 5 (excellent) based on accuracy "
    "and relevance to the ground truth. Respond with ONLY the number."
)


def judge_answer(question: str, answer: str, ground_truth: str) -> float:
    prompt = f"Question: {question}\nGround truth: {ground_truth}\nAnswer to grade: {answer}"
    result = complete(JUDGE_SYSTEM_PROMPT, prompt, provider=JUDGE_PROVIDER)
    try:
        return float(result.text.strip().split()[0])
    except (ValueError, IndexError):
        return 0.0


def main():
    with open(TESTSET_PATH) as f:
        testset = json.load(f)

    providers = ["openai", "gemini", "claude"]
    results = {p: {"latencies": [], "judge_scores": []} for p in providers}

    for provider in providers:
        print(f"\nBenchmarking provider: {provider}")
        for item in testset:
            try:
                answer = ask(item["question"], provider=provider)
            except Exception as e:  # missing key, quota, etc.
                print(f"  skipped ({e})")
                continue
            score = judge_answer(item["question"], answer.text, item["ground_truth"])
            results[provider]["judge_scores"].append(score)

    print("\nProvider benchmark summary")
    print("-" * 50)
    for provider, data in results.items():
        scores = data["judge_scores"]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{provider:<10} avg judge score: {avg:.2f} over {len(scores)} questions")


if __name__ == "__main__":
    main()

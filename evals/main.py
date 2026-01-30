"""
MemLearn Evaluation Framework - Main Entry Point

This script orchestrates the full evaluation pipeline:
1. Ingest dataset conversations into MemLearn agents (if not already done)
2. Query agents with evaluation questions
3. Judge responses and calculate metrics
4. Output results

Usage:
    python -m evals.main --dataset locomo
    python -m evals.main --dataset locomo --sample-id 0 1 2
    python -m evals.main --dataset locomo --skip-ingestion  # If already ingested
    python -m evals.main --dataset locomo --query-only  # Only query, skip ingestion
"""

import argparse
import csv
import json
import os
import sys
import time
import atexit
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memlearn import MemFS, MemLearnConfig, get_memfs_system_prompt_with_note
from memlearn.tools import LangChainToolProvider
from memlearn.types import SessionStatus

from evals.judge import Judge, JudgeResult, QuestionCategory, calculate_metrics

load_dotenv()

# Global variable to track CSV file for emergency cleanup
_GLOBAL_CSV_FILE = None


def _emergency_csv_save():
    """Emergency function to save any pending results if script crashes."""
    global _GLOBAL_CSV_FILE
    if _GLOBAL_CSV_FILE and _GLOBAL_CSV_FILE.exists():
        try:
            # Force flush any pending writes
            with open(_GLOBAL_CSV_FILE, "a", encoding="utf-8") as f:
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass  # Silent failure in emergency cleanup


MODEL_NAME = os.getenv("MEMLEARN_MODEL", "openai:gpt-5.2")
RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset(dataset_name: str) -> list[dict[str, Any]]:
    """Load a dataset by name."""
    if dataset_name == "locomo":
        data_path = Path(__file__).parent / "locomo" / "data" / "locomo10.json"
    elif dataset_name == "longmemeval":
        data_path = Path(__file__).parent / "longmemeval" / "data" / "longmemeval.json"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, "r") as f:
        return json.load(f)


def get_query_system_prompt(agent_name: str, memory_note: str) -> str:
    """Build system prompt for querying the agent."""
    memfs_prompt = get_memfs_system_prompt_with_note(memory_note, extended=True)

    return f"""You are {agent_name}, an AI assistant with persistent memory.

You have previously memorized conversation sessions between two speakers. Now you will be asked questions about those conversations.

**Instructions:**
1. Use your memory tools to search for relevant information
2. Answer questions accurately based on what you have stored
3. If you cannot find the information, say so clearly
4. Be specific and include dates/details when available

{memfs_prompt}

Answer questions concisely but completely. If you need to search your memory, do so before answering."""


def query_agent(
    agent_name: str,
    question: str,
    model_name: str = MODEL_NAME,
    max_tool_rounds: int = 5,
) -> str:
    """Query an existing agent with a question and get its response.

    Uses read-only mode to prevent modifications during retrieval.
    """
    config = MemLearnConfig.default_persistent()
    config.debug = False

    # Use read_only=True for retrieval - no modifications allowed
    memfs = MemFS.for_agent(agent_name, config, read_only=True)
    tool_provider = memfs.get_tool_provider(enable_bash=False)

    model = init_chat_model(model_name, streaming=False)
    tools = tool_provider.get_tools()
    model = model.bind_tools(tools)

    memory_note = memfs.get_memory_note()
    system_prompt = get_query_system_prompt(agent_name, memory_note)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    final_response = ""

    try:
        for _ in range(max_tool_rounds):
            response = model.invoke(messages)
            messages.append(response)

            if response.content:
                final_response = response.content

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                result_str = tool_provider.execute_tool(
                    tool_call["name"],
                    tool_call["args"],
                )
                messages.append(
                    ToolMessage(
                        content=result_str,
                        tool_call_id=tool_call["id"],
                    )
                )

    finally:
        memfs.spindown(status=SessionStatus.COMPLETED)

    return final_response


def get_category_name(category: int) -> str:
    """Get human-readable category name."""
    names = {
        1: "single_hop",
        2: "multi_hop",
        3: "temporal",
        4: "open_domain",
        5: "adversarial",
    }
    return names.get(category, f"category_{category}")


def write_results_csv(
    csv_path: Path, sample_results: list[dict[str, Any]], append_mode: bool = False
) -> None:
    """
    Write detailed evaluation results to a CSV file.

    Args:
        csv_path: Path to CSV file
        sample_results: List of sample results with nested results
        append_mode: If True, append to existing file; if False, create new file

    Columns: sample_id, question_idx, category, category_name, question,
             expected_answer, model_response, is_correct, judge_reasoning, judge_raw_response
    """
    fieldnames = [
        "sample_id",
        "question_idx",
        "category",
        "category_name",
        "question",
        "expected_answer",
        "model_response",
        "is_correct",
        "judge_reasoning",
        "judge_raw_response",
    ]

    mode = "a" if append_mode else "w"
    file_exists = csv_path.exists() and append_mode

    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")

        # Write header only if creating new file or file doesn't exist
        if not file_exists:
            writer.writeheader()

        for sample in sample_results:
            for result in sample.get("results", []):
                writer.writerow(result)


def evaluate_sample(
    sample: dict[str, Any],
    sample_idx: int,
    judge: Judge,
    csv_path: Path | None = None,
    model_name: str = MODEL_NAME,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a single sample by querying the agent and judging responses."""
    sample_id = sample.get("sample_id", sample_idx)
    agent_name = f"locomo-eval-{sample_id}"
    qa_pairs = sample.get("qa", [])

    if verbose:
        print(f"\n{'='*80}")
        print(f"SAMPLE {sample_id} ({len(qa_pairs)} questions)")
        print(f"{'='*80}")

    results = []

    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        expected_answer = qa["answer"]
        category = qa.get("category", 1)
        category_name = get_category_name(category)

        if verbose:
            print(f"\n[{i+1}/{len(qa_pairs)}] Category: {category_name} ({category})")
            print(f"  Question: {question}")
            print(f"  Expected: {expected_answer}")

        try:
            response = query_agent(
                agent_name=agent_name,
                question=question,
                model_name=model_name,
            )
        except Exception as e:
            response = f"Error querying agent: {str(e)}"

        if verbose:
            print(
                f"  Model Response: {response[:300]}{'...' if len(response) > 300 else ''}"
            )

        judge_result = judge.evaluate(
            question=question,
            expected_answer=expected_answer,
            model_response=response,
            category=category,
        )

        results.append(
            {
                "sample_id": sample_id,
                "question_idx": i,
                "question": question,
                "expected_answer": str(expected_answer),
                "model_response": response,
                "category": category,
                "category_name": category_name,
                "is_correct": judge_result.is_correct,
                "judge_reasoning": judge_result.judge_reasoning,
                "judge_raw_response": judge_result.raw_judge_response,
            }
        )

        # Save result to CSV immediately if csv_path provided
        if csv_path:
            try:
                write_results_csv(
                    csv_path,
                    [{"sample_id": sample_id, "results": [results[-1]]}],
                    append_mode=True,
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to save result to CSV: {e}")

        if verbose:
            status = "✓ CORRECT" if judge_result.is_correct else "✗ INCORRECT"
            print(f"  Result: {status}")
            print(
                f"  Judge Response: {judge_result.raw_judge_response[:200]}{'...' if len(judge_result.raw_judge_response) > 200 else ''}"
            )

    correct = sum(1 for r in results if r["is_correct"])
    total = len(results)

    if verbose:
        print(
            f"\n  Sample {sample_id} Summary: {correct}/{total} correct ({correct/total*100:.1f}%)"
        )

    return {
        "sample_id": sample_id,
        "agent_name": agent_name,
        "total_questions": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0.0,
        "results": results,
    }


def run_evaluation(
    dataset_name: str,
    sample_ids: list[int] | None = None,
    skip_ingestion: bool = False,
    model_name: str = MODEL_NAME,
    judge_model: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the full evaluation pipeline.

    Args:
        dataset_name: Name of dataset ('locomo', 'longmemeval')
        sample_ids: Specific sample IDs to evaluate, or None for all
        skip_ingestion: Skip ingestion step (assumes already done)
        model_name: Model for agent queries
        judge_model: Model for judging (defaults to same as model_name)
        verbose: Print progress

    Returns:
        Dictionary with evaluation results and metrics
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name)

    if sample_ids is None:
        sample_ids = list(range(len(dataset)))

    # Initialize CSV file at the start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = RESULTS_DIR / f"{dataset_name}_results_{timestamp}.csv"

    # Register emergency cleanup
    global _GLOBAL_CSV_FILE
    _GLOBAL_CSV_FILE = csv_file
    atexit.register(_emergency_csv_save)

    # Create empty CSV with header
    try:
        write_results_csv(csv_file, [], append_mode=False)
        if verbose:
            print(f"CSV initialized: {csv_file}")
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to initialize CSV file: {e}")
        csv_file = None

    if verbose:
        print(f"MemLearn Evaluation")
        print(f"=" * 60)
        print(f"Dataset: {dataset_name}")
        print(f"Model: {model_name}")
        print(f"Samples: {len(sample_ids)}")
        print(f"Skip ingestion: {skip_ingestion}")
        print()

    if not skip_ingestion:
        if verbose:
            print("Step 1: Ingesting dataset...")

        if dataset_name == "locomo":
            from evals.locomo.run import run_ingestion

            run_ingestion(sample_ids=sample_ids, model_name=model_name)
        else:
            print(f"Warning: No ingestion script for {dataset_name}, skipping...")
    else:
        if verbose:
            print("Step 1: Skipping ingestion (--skip-ingestion)")

    if verbose:
        print("\nStep 2: Evaluating questions...")

    judge = Judge(model_name=judge_model or model_name)

    all_sample_results = []
    all_judge_results = []

    start_time = time.time()

    for idx in sample_ids:
        if idx < 0 or idx >= len(dataset):
            print(f"Warning: Sample index {idx} out of range, skipping")
            continue

        sample = dataset[idx]

        sample_result = evaluate_sample(
            sample=sample,
            sample_idx=idx,
            judge=judge,
            csv_path=csv_file,
            model_name=model_name,
            verbose=verbose,
        )

        all_sample_results.append(sample_result)

        for r in sample_result["results"]:
            all_judge_results.append(
                JudgeResult(
                    question=r["question"],
                    expected_answer=r["expected_answer"],
                    model_response=r["model_response"],
                    category=r["category"],
                    is_correct=r["is_correct"],
                    judge_reasoning=r["judge_reasoning"],
                    raw_judge_response=r.get("judge_raw_response", ""),
                )
            )

    elapsed_time = time.time() - start_time

    if verbose:
        print("\nStep 3: Calculating metrics...")

    metrics = calculate_metrics(all_judge_results)

    final_results = {
        "metadata": {
            "dataset": dataset_name,
            "model": model_name,
            "judge_model": judge_model or model_name,
            "num_samples": len(sample_ids),
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed_time,
        },
        "metrics": metrics,
        "samples": all_sample_results,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"{dataset_name}_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    # Write final CSV (redundant but ensures completeness)
    if csv_file:
        try:
            write_results_csv(csv_file, all_sample_results, append_mode=False)
            if verbose:
                print(f"CSV results saved to: {csv_file}")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to write final CSV: {e}")
    else:
        # Fallback to original behavior if CSV wasn't initialized
        csv_file = RESULTS_DIR / f"{dataset_name}_results_{timestamp}.csv"
        write_results_csv(csv_file, all_sample_results, append_mode=False)
        if verbose:
            print(f"CSV results saved to: {csv_file}")

    if verbose:
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 60}")
        print(f"Total Questions: {metrics['total']}")
        print(f"Correct: {metrics['correct']}")
        print(f"Overall Accuracy: {metrics['accuracy']:.1%}")
        print()
        print("By Category:")
        for cat_name, cat_metrics in metrics.get("by_category", {}).items():
            print(
                f"  {cat_name}: {cat_metrics['correct']}/{cat_metrics['total']} ({cat_metrics['accuracy']:.1%})"
            )
        print()
        print(f"Results saved to: {results_file}")
        print(f"Time elapsed: {elapsed_time:.1f}s")

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Run MemLearn evaluation on a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="locomo",
        choices=["locomo", "longmemeval"],
        help="Dataset to evaluate (default: locomo)",
    )
    parser.add_argument(
        "--sample-id",
        type=int,
        nargs="+",
        help="Specific sample ID(s) to evaluate. If not specified, all samples are evaluated.",
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip the ingestion step (use if data already ingested)",
    )
    parser.add_argument(
        "--query-only",
        action="store_true",
        help="Alias for --skip-ingestion",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Model to use for agent (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model to use for judging (default: same as --model)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    skip_ingestion = args.skip_ingestion or args.query_only

    run_evaluation(
        dataset_name=args.dataset,
        sample_ids=args.sample_id,
        skip_ingestion=skip_ingestion,
        model_name=args.model,
        judge_model=args.judge_model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

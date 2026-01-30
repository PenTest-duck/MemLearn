"""
LoCoMo Dataset Ingestion Script for MemLearn Evaluation.

This script:
1. Creates a new MemLearn instance for each conversation in the locomo dataset
2. Feeds each session (with metadata) as formatted prompts to the agent
3. The agent processes and memorizes the conversation data
4. Logs are stored under the logs folder

Usage:
    python -m evals.locomo.run
    python -m evals.locomo.run --sample-id 0  # Run specific sample
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from memlearn import MemFS, MemLearnConfig, get_memfs_system_prompt_with_note
from memlearn.tools import LangChainToolProvider
from memlearn.types import SessionStatus

load_dotenv()

MODEL_NAME = os.getenv("MEMLEARN_MODEL", "openai:gpt-5.2")
DATA_PATH = Path(__file__).parent / "data" / "locomo10.json"
LOGS_DIR = Path(__file__).parent / "logs"


def setup_logging(sample_id: int) -> logging.Logger:
    """Set up logging for a specific sample run."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"sample_{sample_id}_{timestamp}.log"

    logger = logging.getLogger(f"locomo_sample_{sample_id}")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def load_locomo_dataset() -> list[dict[str, Any]]:
    """Load the LoCoMo dataset from JSON file."""
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def get_system_prompt(agent_name: str, memory_note: str) -> str:
    """Build system prompt for the ingestion agent."""
    memfs_prompt = get_memfs_system_prompt_with_note(memory_note, extended=True)

    return f"""You are {agent_name}, an AI assistant with persistent memory designed to remember conversations for later testing.

Your task is to carefully read and memorize the conversation sessions you are given. Each session contains a dialogue between two speakers with timestamps and metadata.

**Important Instructions:**
1. Read each session carefully and extract key information
2. Store important facts, events, dates, preferences, and relationships in your memory
3. Organize information by speaker and topic for easy retrieval later
4. Pay attention to temporal information (dates, times, sequences of events)
5. Note any personal details, hobbies, relationships, or significant events mentioned

{memfs_prompt}

## Memory Organization Guidelines

- Create files under /memory/ organized by speaker or topic
- Use descriptive filenames like "chris_events.md" or "ryan_family.md"
- Include dates and temporal context in your notes
- Cross-reference related information between speakers
- Summarize key facts in an easily searchable format

Be thorough and systematic in your memorization. This data will be tested later with questions."""


def format_session_prompt(
    session_num: int,
    session_datetime: str,
    session_turns: list[dict],
    speaker_a: str,
    speaker_b: str,
) -> str:
    """Format a session as a prompt for the agent."""
    lines = [
        f"## Session {session_num}",
        f"**Date/Time:** {session_datetime}",
        f"**Speakers:** {speaker_a} and {speaker_b}",
        "",
        "### Conversation:",
        "",
    ]

    for turn in session_turns:
        speaker = turn.get("speaker", "Unknown")
        text = turn.get("text", "")
        dia_id = turn.get("dia_id", "")

        if turn.get("img_url"):
            caption = turn.get("blip_caption", "")
            lines.append(
                f"**{speaker}** [{dia_id}]: [Shared an image: {caption}] {text}"
            )
        else:
            lines.append(f"**{speaker}** [{dia_id}]: {text}")

    lines.append("")
    lines.append(
        "Please read this session carefully and store any important information in your memory."
    )

    return "\n".join(lines)


def get_session_keys(conversation: dict) -> list[tuple[int, str, str]]:
    """Extract session keys from conversation in chronological order."""
    sessions = []

    for key in conversation.keys():
        if key.startswith("session_") and key.endswith("_date_time"):
            continue
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.replace("session_", ""))
                datetime_key = f"{key}_date_time"
                datetime_val = conversation.get(datetime_key, "Unknown")
                sessions.append((num, key, datetime_val))
            except ValueError:
                continue

    sessions.sort(key=lambda x: x[0])
    return sessions


def run_agent_turn(
    model,
    tool_provider: LangChainToolProvider,
    messages: list,
    memfs: MemFS,
    logger: logging.Logger,
    max_tool_rounds: int = 10,
) -> str:
    """Run a single agent turn with tool calling support."""
    final_response = ""

    for round_num in range(max_tool_rounds):
        response = model.invoke(messages)
        messages.append(response)

        if response.content:
            final_response = response.content
            logger.debug(f"Agent response: {response.content[:500]}...")

        if memfs:
            memfs.append_conversation_message(
                {"role": "assistant", "content": response.content or ""}
            )

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call["args"]

            logger.debug(f"Tool call: {tool_name}({json.dumps(arguments)[:200]})")

            result_str = tool_provider.execute_tool(tool_name, arguments)

            try:
                result_data = json.loads(result_str)
                status = result_data.get("status", "unknown")
                logger.debug(f"Tool result: {status}")
            except json.JSONDecodeError:
                pass

            messages.append(
                ToolMessage(
                    content=result_str,
                    tool_call_id=tool_call["id"],
                )
            )

    return final_response


def ingest_sample(
    sample: dict[str, Any],
    sample_idx: int,
    logger: logging.Logger,
    model_name: str = MODEL_NAME,
) -> dict[str, Any]:
    """Ingest a single sample (conversation) into MemLearn."""
    sample_id = sample.get("sample_id", sample_idx)
    conversation = sample.get("conversation", {})

    speaker_a = conversation.get("speaker_a", "Speaker A")
    speaker_b = conversation.get("speaker_b", "Speaker B")

    logger.info(f"Starting ingestion for sample {sample_id}")
    logger.info(f"Speakers: {speaker_a} and {speaker_b}")

    agent_name = f"locomo-eval-{sample_id}"

    config = MemLearnConfig.default_persistent()
    config.debug = False

    # Ingestion uses read_only=False (default) for full read-write access
    memfs = MemFS.for_agent(agent_name, config, read_only=False)
    tool_provider = memfs.get_tool_provider(enable_bash=False)

    model = init_chat_model(model_name, streaming=False)
    tools = tool_provider.get_tools()
    model = model.bind_tools(tools)

    memory_note = memfs.get_memory_note()
    system_prompt = get_system_prompt(agent_name, memory_note)
    messages = [SystemMessage(content=system_prompt)]

    sessions = get_session_keys(conversation)
    logger.info(f"Found {len(sessions)} sessions to process")

    stats = {
        "sample_id": sample_id,
        "agent_name": agent_name,
        "num_sessions": len(sessions),
        "sessions_processed": 0,
        "start_time": time.time(),
        "end_time": None,
        "status": "running",
    }

    try:
        for session_num, session_key, session_datetime in sessions:
            session_turns = conversation.get(session_key, [])

            if not session_turns:
                logger.warning(f"Session {session_num} is empty, skipping")
                continue

            logger.info(
                f"Processing session {session_num} ({len(session_turns)} turns)"
            )

            prompt = format_session_prompt(
                session_num=session_num,
                session_datetime=session_datetime,
                session_turns=session_turns,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
            )

            messages.append(HumanMessage(content=prompt))
            memfs.append_conversation_message({"role": "user", "content": prompt})

            response = run_agent_turn(
                model=model,
                tool_provider=tool_provider,
                messages=messages,
                memfs=memfs,
                logger=logger,
            )

            stats["sessions_processed"] += 1
            logger.info(f"Completed session {session_num}")

        stats["status"] = "completed"

    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        stats["status"] = "error"
        stats["error"] = str(e)

    finally:
        stats["end_time"] = time.time()
        stats["duration_seconds"] = stats["end_time"] - stats["start_time"]

        memfs.spindown(
            status=(
                SessionStatus.COMPLETED
                if stats["status"] == "completed"
                else SessionStatus.ABORTED
            )
        )

        logger.info(f"Ingestion complete for sample {sample_id}")
        logger.info(f"Duration: {stats['duration_seconds']:.2f}s")
        logger.info(
            f"Sessions processed: {stats['sessions_processed']}/{stats['num_sessions']}"
        )

    return stats


def run_ingestion(
    sample_ids: list[int] | None = None,
    model_name: str = MODEL_NAME,
) -> list[dict[str, Any]]:
    """Run ingestion for specified samples or all samples."""
    dataset = load_locomo_dataset()

    if sample_ids is None:
        sample_ids = list(range(len(dataset)))

    print(f"LoCoMo Dataset Ingestion")
    print(f"=" * 50)
    print(f"Model: {model_name}")
    print(f"Samples to process: {len(sample_ids)}")
    print(f"Logs directory: {LOGS_DIR}")
    print()

    all_stats = []

    for idx in sample_ids:
        if idx < 0 or idx >= len(dataset):
            print(f"Warning: Sample index {idx} out of range, skipping")
            continue

        sample = dataset[idx]
        logger = setup_logging(idx)

        print(f"\n[{idx + 1}/{len(sample_ids)}] Processing sample {idx}...")

        stats = ingest_sample(
            sample=sample,
            sample_idx=idx,
            logger=logger,
            model_name=model_name,
        )

        all_stats.append(stats)

        print(f"  Status: {stats['status']}")
        print(f"  Sessions: {stats['sessions_processed']}/{stats['num_sessions']}")
        print(f"  Duration: {stats.get('duration_seconds', 0):.2f}s")

    summary_file = (
        LOGS_DIR / f"ingestion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Ingestion Complete")
    print(f"Summary saved to: {summary_file}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Ingest LoCoMo dataset into MemLearn")
    parser.add_argument(
        "--sample-id",
        type=int,
        nargs="+",
        help="Specific sample ID(s) to process. If not specified, all samples are processed.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Model to use (default: {MODEL_NAME})",
    )

    args = parser.parse_args()

    run_ingestion(
        sample_ids=args.sample_id,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Behavioral eval harness for repo-arch LoRA adapter.

Usage:
  # Sanity run (10 questions, all modes)
  python3 scripts/eval_behavior.py \
    --questions .repo-arch/eval/questions.jsonl \
    --limit 10 \
    --modes base retrieval lora \
    --out .repo-arch/eval/runs/sanity-10.jsonl

  # Full run
  python3 scripts/eval_behavior.py \
    --questions .repo-arch/eval/questions.jsonl \
    --modes base retrieval lora \
    --out .repo-arch/eval/runs/full-45x3.jsonl

  # Single mode
  python3 scripts/eval_behavior.py \
    --questions .repo-arch/eval/questions.jsonl \
    --modes retrieval \
    --out .repo-arch/eval/runs/retrieval-only.jsonl

Requires: source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ── config ────────────────────────────────────────────────────────
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAPTER_DIR = os.path.join(REPO_DIR, ".repo-arch", "adapters", "repo-arch-40c05f5")
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a repo-aware coding assistant for the pi-monorepo. "
    "Answer based on the repository's git history and architecture. "
    "If you are not sure, say so. Be specific about file paths and packages."
)


def load_questions(path, limit=None):
    questions = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            if q["id"] <= 0:
                continue
            questions.append(q)
    if limit:
        # first `limit` or evenly spaced? We'll take first `limit` for reproducibility.
        questions = questions[:limit]
    return questions


# ── inference helpers ─────────────────────────────────────────────


def infer_base_model(question, max_tokens=512):
    """Run base model (no adapter) via mlx_lm.generate"""
    user_prompt = format_user_prompt(question, context=None)
    return run_mlx_generate(SYSTEM_PROMPT, user_prompt, adapter_path=None, max_tokens=max_tokens)


def infer_lora(question, max_tokens=512):
    """Run LoRA adapter model"""
    user_prompt = format_user_prompt(question, context=None)
    return run_mlx_generate(SYSTEM_PROMPT, user_prompt, adapter_path=ADAPTER_DIR, max_tokens=max_tokens)


def infer_retrieval(question, max_tokens=512):
    """Retrieval-only: query repo-arch, format context, feed into base model"""
    # Step 1: retrieve cards via repo-arch similar
    cards = query_repo_arch(question)
    context = format_cards_as_context(cards) if cards else "No relevant historical data found."

    # Step 2: answer with context using base model
    user_prompt = format_user_prompt(question, context=context)
    return run_mlx_generate(SYSTEM_PROMPT, user_prompt, adapter_path=None, max_tokens=max_tokens)


# ── core functions ────────────────────────────────────────────────


def format_user_prompt(question, context=None):
    """Format the user prompt, optionally with context."""
    if context:
        return (
            f"Context from repo history:\n{context}\n\n"
            f"Question: {question}"
        )
    return question


def run_mlx_generate(system_prompt, user_prompt, adapter_path=None, max_tokens=512, temperature=0.3):
    """Run mlx_lm.generate via subprocess with the correct CLI syntax."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", MODEL_NAME,
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
        "--system-prompt", system_prompt,
        "--prompt", user_prompt,
        "--use-default-chat-template",
    ]
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])

    try:
        start = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=REPO_DIR
        )
        elapsed = time.time() - start
        output = result.stdout.strip()
        # Parse answer: everything between ========== markers after "Generation"
        parts = output.split("==========")
        if len(parts) >= 3:
            answer = parts[-2].strip() if parts[-2].strip() else parts[-3].strip()
        else:
            answer = output
        # Filter out stats line like "Prompt: 36 tokens, ..."
        stats_lines = [l for l in answer.split("\n") if "tokens-per-sec" in l or "Peak memory" in l]
        for sl in stats_lines:
            answer = answer.replace(sl, "")
        answer = answer.strip()
        return {
            "answer": answer,
            "latency": round(elapsed, 2),
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {"answer": None, "latency": 120, "error": "timeout"}
    except Exception as e:
        return {"answer": None, "latency": 0, "error": str(e)}


def query_repo_arch(question):
    """Query repo-arch similar command for relevant cards."""
    cmd = ["repo-arch", "similar", question, "--json", "--repo", REPO_DIR]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=REPO_DIR)
        if result.returncode != 0:
            print(f"  [WARN] repo-arch similar failed: {result.stderr[:200]}", file=sys.stderr)
            return None
        data = json.loads(result.stdout)
        cards = data.get("results", [])
        # Filter low-confidence results
        cards = [c for c in cards if c.get("score", 0) > 0.1]
        return cards[:3]  # top 3 cards
    except Exception as e:
        print(f"  [WARN] repo-arch similar error: {e}", file=sys.stderr)
        return None


def format_cards_as_context(cards):
    """Format retrieved cards into a readable context block."""
    parts = []
    for i, card in enumerate(cards, 1):
        text = card.get("text", "")
        score = card.get("score", 0)
        meta = card.get("metadata", {})
        status = meta.get("status", "unknown")
        ctype = meta.get("type", "unknown")
        parts.append(
            f"[Card {i}] (type={ctype}, status={status}, score={score:.3f})\n"
            f"{text[:600]}"
        )
    return "\n\n".join(parts)


# ── main loop ─────────────────────────────────────────────────────


def run_eval(questions, modes, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = []
    total = len(questions) * len(modes)
    done = 0

    for q in questions:
        qid = q["id"]
        question = q["question"]

        for mode in modes:
            done += 1
            print(f"[{done}/{total}] Q{qid} mode={mode} ... ", end="", flush=True)

            if mode == "base":
                out = infer_base_model(question)
            elif mode == "lora":
                out = infer_lora(question)
            elif mode == "retrieval":
                out = infer_retrieval(question)
            else:
                out = {"answer": None, "latency": 0, "error": f"unknown mode: {mode}"}

            result = {
                "question_id": qid,
                "question": question,
                "mode": mode,
                "answer": out.get("answer"),
                "latency": out.get("latency"),
                "error": out.get("error"),
            }
            results.append(result)

            if out.get("error"):
                print(f"ERROR: {out['error'][:100]}")
            else:
                print(f"ok ({out['latency']}s)")

    # Write output
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone. {len(results)} results written to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Behavioral eval for repo-arch LoRA")
    parser.add_argument("--questions", default=".repo-arch/eval/questions.jsonl",
                        help="Path to questions JSONL")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate first N questions (sanity check)")
    parser.add_argument("--modes", nargs="+", default=["base", "retrieval", "lora"],
                        choices=["base", "retrieval", "lora"],
                        help="Which modes to evaluate")
    parser.add_argument("--out", default=".repo-arch/eval/runs/full.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()

    questions_path = os.path.join(REPO_DIR, args.questions)
    output_path = os.path.join(REPO_DIR, args.out)

    questions = load_questions(questions_path, limit=args.limit)
    print(f"Loaded {len(questions)} questions, modes={args.modes}")
    print(f"Output: {output_path}")

    run_eval(questions, args.modes, output_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hybrid answer: given a question, retrieve cards, format context, and answer with LoRA.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/hybrid_answer.py "What should I know about agent-session.ts?"
  python3 scripts/hybrid_answer.py --question "..." --adapter .repo-arch/adapters/repo-arch-b8125c4
  python3 scripts/hybrid_answer.py --json --question "..."
"""

import argparse
import json
import os
import subprocess
import sys
import time

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADAPTER_DIR = os.path.join(REPO_DIR, ".repo-arch", "adapters", "hybrid-lora")
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a repo-aware coding assistant. "
    "Use the repo-arch cards as your only source of repo knowledge. "
    "Be specific about file paths, commit counts, and package names. "
    "If the cards do not contain the answer, say what information is missing."
)


def load_cards(repo_dir):
    """Load cached cards."""
    import glob as _glob
    card_files = _glob.glob(os.path.join(repo_dir, ".repo-arch", "cache", "cards", "*.json"))
    if not card_files:
        return []
    with open(card_files[0]) as f:
        data = json.load(f)
    cards = data.get("cards", [])

    review_path = os.path.join(repo_dir, ".repo-arch", "review-state.json")
    if os.path.exists(review_path):
        with open(review_path) as f:
            review = json.load(f)
    else:
        review = {}

    for c in cards:
        cid = c.get("id", "")
        c["status"] = review.get(cid, {}).get("status", "unreviewed")
    return cards


def serialize_card(card, status="unknown"):
    """Serialize a card into the canonical <CARD> format."""
    ctype = card.get("type", "unknown")
    confidence = card.get("confidence", 0)
    affected = card.get("affectedFiles", [])
    suggestion = card.get("suggestion", "")
    title = card.get("title", "")

    lines = [f'<CARD type="{ctype}" status="{status}" confidence="{confidence}">']
    if affected:
        lines.append("Affected files:")
        for f in affected:
            lines.append(f"  {f}")
    summary = suggestion[:600] if suggestion else title
    lines.append(f"Summary: {summary}")
    lines.append("</CARD>")
    return "\n".join(lines)


def query_repo_arch(question, repo_dir, top_k=5):
    """Retrieve relevant cards via repo-arch similar."""
    try:
        result = subprocess.run(
            ["repo-arch", "similar", question, "--json", "--repo", repo_dir],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        cards = data.get("results", [])
        # Filter low-confidence
        return [c for c in cards if c.get("score", 0) > 0.1][:top_k]
    except Exception as e:
        print(f"  [WARN] similar query failed: {e}", file=sys.stderr)
        return []


def build_context(question, cards, card_map, max_cards=5):
    """Build context from retrieved cards."""
    context_cards = []
    seen_ids = set()

    for result in cards:
        cid = result.get("id", "")
        if cid in card_map and cid not in seen_ids:
            card = card_map[cid]
            status = card.get("status", "unknown")
            if status == "accepted":
                context_cards.append(serialize_card(card, status=status))
                seen_ids.add(cid)

    return context_cards[:max_cards]


def answer_with_model(context_str, question, adapter_path=None, max_tokens=512, temperature=0.3):
    """Call MLX model with system prompt + user prompt."""
    user_prompt = (
        "Use the following repo-arch cards as context. "
        "Answer based only on the information in these cards. "
        "Be specific about file paths, commit counts, and package names. "
        "If the cards do not contain the answer, say what information is missing.\n\n"
        f"### Context Cards\n{context_str}\n\n"
        f"### Question\n{question}"
    )

    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", MODEL_NAME,
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
        "--system-prompt", SYSTEM_PROMPT,
        "--prompt", user_prompt,
        "--use-default-chat-template",
    ]
    if adapter_path and os.path.exists(adapter_path):
        cmd.extend(["--adapter-path", adapter_path])

    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start

        output = result.stdout.strip()
        parts = output.split("==========")
        if len(parts) >= 3:
            answer = parts[-2].strip() if parts[-2].strip() else parts[-3].strip()
        else:
            answer = output
        # Strip stats lines
        for line in answer.split("\n"):
            if "tokens-per-sec" in line or "Peak memory" in line or "Prompt:" in line:
                answer = answer.replace(line, "")
        answer = answer.strip()
        return {"answer": answer, "latency": round(elapsed, 2), "error": None}
    except subprocess.TimeoutExpired:
        return {"answer": None, "latency": 120, "error": "timeout"}
    except Exception as e:
        return {"answer": None, "latency": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Hybrid answer with retrieval+LoRA")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--question", "-q", dest="qflag", help="Alternative question flag")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context cards")
    parser.add_argument("--base", action="store_true", help="Use base model without adapter")
    args = parser.parse_args()

    question = args.question or args.qflag
    if not question:
        parser.print_help()
        sys.exit(1)

    adapter_path = args.adapter
    if not adapter_path and not args.base:
        # Try default hybrid adapter path
        default_adapter = os.path.join(REPO_DIR, ".repo-arch", "adapters", "hybrid-lora")
        if os.path.exists(default_adapter):
            adapter_path = default_adapter
        else:
            print("No adapter found; using base model. Train one with hybrid_export.py first.", file=sys.stderr)

    # ── Retrieve context ──
    cards = load_cards(REPO_DIR)
    card_map = {c["id"]: c for c in cards}

    similar_results = query_repo_arch(question, REPO_DIR, top_k=args.top_k)
    context_cards = build_context(question, similar_results, card_map, max_cards=args.top_k)

    if context_cards:
        context_str = "\n\n".join(context_cards)
        context_source = [c.split("\n")[0] for c in context_cards]
    else:
        context_str = "No relevant repo-arch cards found for this question."
        context_source = []

    # ── Answer ──
    result = answer_with_model(
        context_str, question,
        adapter_path=adapter_path if not args.base else None,
    )

    if args.json:
        output = {
            "question": question,
            "context_cards": context_source,
            "answer": result.get("answer"),
            "latency": result.get("latency"),
            "error": result.get("error"),
            "mode": "base" if args.base else "hybrid",
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  Question: {question}")
        print(f"{'='*60}")
        if context_source:
            print(f"\n  Context ({len(context_cards)} cards):")
            for s in context_source:
                print(f"    - {s}")
        print(f"\n  Answer ({result['latency']}s):")
        print(f"  {result.get('answer', 'ERROR: ' + str(result.get('error')))}")
        print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Hybrid export: generate context+question->answer training data.

Takes the repo-arch card set and generates a training dataset where
each example includes card context, so the adapter learns to use
cards to answer questions rather than memorizing facts.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/hybrid_export.py --repo . --out .repo-arch/training-data/hybrid/

Output:
  train.jsonl  (context + question -> answer)
  valid.jsonl  (hold-out examples)
  negative.jsonl (negative/refusal examples)
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys

random.seed(42)

# ── card serialization ───────────────────────────────────────────


def serialize_card(card, status="unknown"):
    """Serialize a card into the canonical <CARD> format."""
    ctype = card.get("type", "unknown")
    confidence = card.get("confidence", 0)
    affected = card.get("affectedFiles", [])
    suggestion = card.get("suggestion", "")

    lines = [f'<CARD type="{ctype}" status="{status}" confidence="{confidence}">']

    if affected:
        lines.append("Affected files:")
        for f in affected:
            lines.append(f"  {f}")

    # Extract key info from suggestion (first 300 chars)
    summary = suggestion[:500] if suggestion else card.get("title", "")
    lines.append(f"Summary: {summary}")

    # Supporting commits
    commits = card.get("supportingCommits", [])
    if commits:
        lines.append("Notable commits:")
        for c in commits[:5]:  # top 5 only
            lines.append(f"  {c['sha'][:12]}: {c['subject'][:100]}")

    lines.append("</CARD>")
    return "\n".join(lines)


def card_to_qa_templates(card, status):
    """Generate question templates for a card."""
    ctype = card.get("type", "")
    affected = card.get("affectedFiles", [])
    suggestion = card.get("suggestion", "")[:500]
    title = card.get("title", "")

    templates = []

    if ctype == "co-change":
        files_str = ", ".join(affected[:3])
        templates.extend([
            {
                "question": f"What files tend to change together in this project?",
                "answer": f"The following files co-change: {files_str}. {suggestion}",
            },
            {
                "question": f"How are {affected[0] if affected else ''} and {affected[1] if len(affected) > 1 else ''} related?",
                "answer": suggestion,
            },
        ])
    elif ctype == "repeated-fix":
        file_str = affected[0] if affected else title
        templates.extend([
            {
                "question": f"What should I know before modifying {file_str}?",
                "answer": f"Warning: {suggestion}",
            },
            {
                "question": f"What keeps breaking in {file_str}?",
                "answer": f"Repeated fixes: {suggestion}",
            },
        ])
    elif ctype == "test-gap":
        file_str = affected[0] if affected else title
        templates.extend([
            {
                "question": f"Is there test coverage for {file_str}?",
                "answer": f"Possible test gap: {suggestion}",
            },
            {
                "question": f"How many times has {file_str} changed without tests?",
                "answer": suggestion,
            },
        ])
    elif ctype == "revert-pattern":
        file_str = affected[0] if affected else title
        templates.extend([
            {
                "question": f"Which files have been reverted most often?",
                "answer": f"Reversion pattern: {suggestion}",
            },
            {
                "question": f"Has {file_str} been unstable historically?",
                "answer": f"Yes, reversion history: {suggestion}",
            },
        ])
    elif ctype == "high-churn":
        file_str = affected[0] if affected else title
        templates.extend([
            {
                "question": f"Which files have the highest change frequency?",
                "answer": f"High-churn hotspot: {suggestion}",
            },
            {
                "question": f"How often does {file_str} change?",
                "answer": suggestion,
            },
        ])

    return templates


def hybrid_format(context_cards, question, answer):
    """Format as context + question -> answer with chat template."""
    context_str = "\n\n".join(context_cards) if context_cards else "No relevant historical data found in repo-arch cards."
    messages = [
        {
            "role": "user",
            "content": (
                "Use the following repo-arch cards as context. "
                "Answer based only on the information in these cards. "
                "Be specific about file paths, commit counts, and package names. "
                "If the cards do not contain the answer, say what information is missing.\n\n"
                f"### Context Cards\n{context_str}\n\n"
                f"### Question\n{question}"
            ),
        },
        {"role": "assistant", "content": answer},
    ]
    return {"messages": messages}


def load_cards(repo_dir):
    """Load cards from repo-arch cache and review state."""
    card_files = glob.glob(os.path.join(repo_dir, ".repo-arch", "cache", "cards", "*.json"))
    if not card_files:
        print("No card cache found. Run `repo-arch cards` first.", file=sys.stderr)
        return []

    with open(card_files[0]) as f:
        data = json.load(f)
    cards = data.get("cards", [])

    # Map card IDs -> cards
    card_map = {c["id"]: c for c in cards}

    # Load review state
    review_path = os.path.join(repo_dir, ".repo-arch", "review-state.json")
    if os.path.exists(review_path):
        with open(review_path) as f:
            review = json.load(f)
    else:
        review = {}

    # Annotate with status
    for c in cards:
        cid = c.get("id", "")
        if cid in review:
            c["status"] = review[cid]["status"]
        else:
            c["status"] = "unreviewed"

    return cards, card_map, review


def find_similar_cards(query, repo_dir, top_k=5):
    """Use repo-arch similar to find related cards."""
    try:
        result = subprocess.run(
            ["repo-arch", "similar", query, "--json", "--repo", repo_dir],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return data.get("results", [])[:top_k]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Hybrid dataset export")
    parser.add_argument("--repo", default=".", help="Repo path")
    parser.add_argument("--out", default=".repo-arch/training-data/hybrid", help="Output dir")
    parser.add_argument("--split-valid", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--examples-per-card", type=int, default=2, help="Max examples per card")
    parser.add_argument("--negative-ratio", type=float, default=0.2, help="Fraction of negative examples")
    parser.add_argument("--context-cards", type=int, default=4, help="Max context cards per example")
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo)
    out_dir = os.path.join(repo_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    cards, card_map, review = load_cards(repo_dir)

    accepted = [c for c in cards if c.get("status") == "accepted"]
    rejected = [c for c in cards if c.get("status") == "rejected"]

    print(f"Loaded {len(cards)} cards ({len(accepted)} accepted, {len(rejected)} rejected)")

    # ── Generate positive examples ──
    all_examples = []
    for card in accepted:
        serialized = serialize_card(card, status="accepted")
        templates = card_to_qa_templates(card, "accepted")

        for tmpl in templates[:args.examples_per_card]:
            # Find similar cards for context
            similar = find_similar_cards(tmpl["question"], repo_dir, top_k=args.context_cards)

            # Build context: source card + similar accepted cards
            context_cards = [serialized]
            for s in similar:
                sid = s.get("id", "")
                if sid in card_map and sid != card.get("id"):
                    sc = card_map[sid]
                    if sc.get("status") == "accepted":
                        context_cards.append(serialize_card(sc, status="accepted"))

            # Limit context to args.context_cards
            context_cards = context_cards[:args.context_cards]

            example = hybrid_format(context_cards, tmpl["question"], tmpl["answer"])
            example["source_card"] = card.get("id", "")
            example["card_type"] = card.get("type", "")
            all_examples.append(example)

    print(f"Generated {len(all_examples)} positive examples")

    # ── Generate negative examples ──
    # Use questions about files NOT covered by any card
    uncovered_questions = [
        "What are the test patterns for the TUI overlay system?",
        "How should I structure a new provider extension?",
        "What is the performance profile of the markdown renderer?",
        "How does session compaction interact with branched sessions?",
        "What is the best approach for adding a new slash command?",
        "How are extension resources discovered and cached?",
        "What is the architecture of the event bus system?",
        "How does theme inheritance work for custom extensions?",
        "What is the startup sequence and initialization order?",
        "How does the OAuth token refresh mechanism work?",
    ]

    negative_count = int(len(all_examples) * args.negative_ratio / (1 - args.negative_ratio))
    selected_negatives = random.sample(uncovered_questions, min(negative_count, len(uncovered_questions)))

    negative_examples = []
    for q in selected_negatives:
        # Try to find similar cards first
        similar = find_similar_cards(q, repo_dir, top_k=3)
        if similar:
            # If cards exist but none match, that's still a negative
            context_cards = []
            for s in similar[:2]:
                sid = s.get("id", "")
                if sid in card_map:
                    context_cards.append(serialize_card(card_map[sid]))
        else:
            context_cards = []

        # For true negatives, no cards match or cards don't contain the answer
        answer = (
            "The available repo-arch cards do not contain information about this topic. "
            "The cards cover co-change patterns, repeated fixes, test gaps, reversion patterns, "
            "and high-churn files in specific packages. This question is outside that coverage."
        )
        example = hybrid_format(context_cards, q, answer)
        example["source_card"] = "none"
        example["card_type"] = "negative"
        negative_examples.append(example)

    print(f"Generated {len(negative_examples)} negative examples")

    # ── Split and write ──
    random.shuffle(all_examples)
    valid_count = max(1, int(len(all_examples) * args.split_valid))
    train_examples = all_examples[valid_count:]
    valid_examples = all_examples[:valid_count]

    # Add some negatives to both splits
    n_neg_valid = max(0, len(valid_examples) // 2)
    n_neg_train = len(negative_examples) - n_neg_valid

    valid_examples += negative_examples[:n_neg_valid]
    train_examples += negative_examples[n_neg_valid:n_neg_valid + n_neg_train]

    random.shuffle(train_examples)
    random.shuffle(valid_examples)

    def write_jsonl(path, examples):
        with open(path, "w") as f:
            for ex in examples:
                # Strip non-message metadata
                msg = {"messages": ex["messages"]}
                f.write(json.dumps(msg) + "\n")
        print(f"  {path}: {len(examples)} examples ({os.path.getsize(path)} bytes)")

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_examples)
    write_jsonl(os.path.join(out_dir, "valid.jsonl"), valid_examples)

    # Also write a metadata file
    meta = {
        "total_cards": len(cards),
        "accepted_cards": len(accepted),
        "rejected_cards": len(rejected),
        "positive_examples": len(all_examples),
        "negative_examples": len(negative_examples),
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "context_cards_per_example": args.context_cards,
        "examples_per_card": args.examples_per_card,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {json.dumps(meta, indent=2)}")

    # Show a sample
    print(f"\n=== Sample Example ===")
    sample = all_examples[0] if all_examples else train_examples[0]
    print(json.dumps(sample, indent=2)[:1000])


if __name__ == "__main__":
    main()

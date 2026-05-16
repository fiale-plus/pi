#!/usr/bin/env python3
"""
Large-scale synthetic dataset generator for overnight LoRA training.

Strategy:
  1. For each card and commit cluster, retrieve context (cards + commits)
  2. Use base model to generate a high-quality answer from that context
  3. Use (context + question -> answer) as training example
  4. Include distractors, negatives, and edge cases

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/gen_large_dataset.py --target 1000 --out .repo-arch/training-data/large
"""

import argparse
import glob
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

random.seed(42)

# ── BM25 (same impl as before) ───────────────────────────────────


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.documents = []; self.doc_id_to_idx = {}; self.idx_to_doc_id = {}
        self.avgdl = 0; self.N = 0; self.df = Counter(); self.term_in_docs = defaultdict(set)

    def tokenize(self, text):
        return re.findall(r'[a-zA-Z][a-zA-Z0-9_\-\./]{2,}', text.lower())

    def add_document(self, doc_id, text):
        idx = len(self.documents)
        self.doc_id_to_idx[doc_id] = idx
        self.idx_to_doc_id[idx] = doc_id
        tokens = self.tokenize(text)
        self.documents.append(tokens)
        for term in set(tokens):
            self.df[term] += 1; self.term_in_docs[term].add(idx)
        self.N = len(self.documents)
        self.avgdl = sum(len(d) for d in self.documents) / max(1, self.N)

    def idf(self, term):
        n = self.df.get(term, 0)
        return 0 if n == 0 else math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query, top_k=20):
        tokens = self.tokenize(query)
        if not tokens: return []
        scores = defaultdict(float)
        for term in tokens:
            idf_val = self.idf(term)
            if idf_val == 0: continue
            for doc_idx in self.term_in_docs.get(term, set()):
                dl = len(self.documents[doc_idx])
                tf = self.documents[doc_idx].count(term)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf_val * tf * (self.k1 + 1) / max(denom, 0.001)
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return [(self.idx_to_doc_id.get(doc_idx, ""), score) for doc_idx, score in sorted_scores[:top_k]]


# ── Load indexes ─────────────────────────────────────────────────


def load_card_index(repo_dir):
    cfiles = glob.glob(os.path.join(repo_dir, ".repo-arch", "cache", "cards", "*.json"))
    if not cfiles: return None, None, None
    with open(cfiles[0]) as f: data = json.load(f)
    cards = data.get("cards", [])
    rev_path = os.path.join(repo_dir, ".repo-arch", "review-state.json")
    review = json.load(open(rev_path)) if os.path.exists(rev_path) else {}
    bm25 = BM25()
    for c in cards:
        cid = c.get("id", "")
        c["status"] = review.get(cid, {}).get("status", "unreviewed")
        text = f"{c.get('title','')} {c.get('suggestion','')} {' '.join(c.get('affectedFiles',[]))}"
        bm25.add_document(cid, text)
    return cards, bm25, review


def load_commit_index(repo_dir):
    idx_path = os.path.join(repo_dir, ".repo-arch", "index", "full", "index.json")
    if not os.path.exists(idx_path): return None, None
    with open(idx_path) as f: data = json.load(f)
    docs = data.get("documents", [])
    seen = set(); unique = []
    for d in docs:
        if d.get("sha","") not in seen:
            seen.add(d["sha"]); unique.append(d)
    docs = unique
    bm25 = BM25()
    for doc in docs: bm25.add_document(doc["id"], doc["text"])
    return docs, bm25


def format_card(card, include_commits=True):
    """One-line card summary."""
    affected = card.get("affectedFiles", [])
    files = ", ".join(affected[:3])
    suggestion = card.get("suggestion", "")[:300]
    lines = [
        f"CARD ({card.get('type','')}) [{card.get('status','')}] confidence={card.get('confidence',0)}",
        f"  Files: {files}",
        f"  Summary: {suggestion}",
    ]
    return "\n".join(lines)


def format_commit(doc):
    """One-line commit summary."""
    pkgs = ", ".join(doc.get("packages", [])[:3])
    files = doc.get("paths", [])[:4]
    lines = [
        f"COMMIT {doc.get('sha','')}",
        f"  Subject: {doc.get('subject','')[:200]}",
    ]
    if pkgs: lines.append(f"  Packages: {pkgs}")
    if files: lines.append(f"  Files: {', '.join(files)}")
    return "\n".join(lines)


# ── Generate questions ───────────────────────────────────────────


def generate_questions_for_card(card, count=8):
    """Generate diverse questions for a card pattern."""
    ctype = card.get("type", "")
    affected = card.get("affectedFiles", [])
    suggestions = card.get("suggestion", "")[:200]

    questions = []
    if ctype == "co-change":
        files_str = ", ".join(affected[:3])
        for f in affected[:2]:
            questions.append(f"Which files co-change with {f}?")
            questions.append(f"What should I check when modifying {f}?")
        questions.append(f"What does the co-change between {files_str} imply?")
        questions.append("How tightly coupled are the files that co-change together?")

    elif ctype == "repeated-fix":
        for f in affected[:2]:
            questions.append(f"What should I know before modifying {f}?")
            questions.append(f"How many times has {f} been fixed?")
            questions.append(f"What keeps breaking in {f}?")
        questions.append("Which files have the most bug fix churn?")

    elif ctype == "test-gap":
        for f in affected[:2]:
            questions.append(f"Is there test coverage for {f}?")
            questions.append(f"How many changes to {f} lack test updates?")
        questions.append("Which files changed most without corresponding tests?")

    elif ctype == "revert-pattern":
        for f in affected[:2]:
            questions.append(f"Has {f} been unstable historically?")
        questions.append("Which files have been reverted most often?")

    elif ctype == "high-churn":
        for f in affected[:2]:
            questions.append(f"How often does {f} change?")
            questions.append(f"What is the churn rate of {f}?")
        questions.append("Which files change most frequently in this repo?")

    # General questions from any card type
    for f in affected[:2]:
        questions.append(f"What is the history of changes to {f}?")
        questions.append(f"What risks are associated with {f}?")

    return list(set(questions))[:count]


SYSTEM_PROMPT_GEN = (
    "You are a repo-analysis assistant. Given context from a repository, "
    "answer the question concisely. Cite specific file paths, commit SHAs, "
    "and numbers. If the context does not contain the answer, say so."
)


def generate_answer(context_str, question, max_tokens=384, temperature=0.2):
    """Use base model to generate a reference answer from context."""
    user_prompt = f"### Context\n{context_str}\n\n### Question\n{question}\n\n### Answer"
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
        "--system-prompt", "Answer concisely using the context above. Cite specific evidence.",
        "--prompt", user_prompt,
        "--use-default-chat-template",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout.strip()
        parts = output.split("==========")
        answer = parts[-2].strip() if len(parts) >= 3 and parts[-2].strip() else parts[-3].strip() if len(parts) >= 3 else output
        for line in answer.split("\n"):
            if "tokens-per-sec" in line or "Peak memory" in line or "Prompt:" in line:
                answer = answer.replace(line, "")
        return answer.strip() or "[empty response]"
    except Exception as e:
        return f"[generation error: {e}]"


def build_context(cards, card_bm25, commits, commit_bm25, question, max_cards=3, max_commits=6):
    """Build fused context for a question. Includes distractors."""
    context_parts = ["--- REPO-ARCH CARDS ---"]

    # Relevant cards
    card_results = card_bm25.search(question, top_k=max_cards * 3) if card_bm25 else []
    added = 0
    seen = set()
    for cid, score in card_results:
        card = next((c for c in cards if c["id"] == cid), None) if cards else None
        if not card or card.get("status") == "rejected" or card.get("type","") in seen:
            continue
        seen.add(card.get("type",""))
        context_parts.append(format_card(card))
        added += 1
        if added >= max_cards:
            break

    # Add 1-2 distractor cards (random accepted cards not matching query)
    if cards and len(cards) > max_cards + 2:
        accepted = [c for c in cards if c.get("status") == "accepted" and c.get("type","") not in seen]
        distractor_count = random.randint(1, 2)
        for c in random.sample(accepted, min(distractor_count, len(accepted))):
            context_parts.append(format_card(c))
            seen.add(c.get("type",""))

    context_parts.append("\n--- RELEVANT COMMITS ---")
    commit_results = commit_bm25.search(question, top_k=max_commits * 2) if commit_bm25 else []
    seen_sha = set()
    for did, score in commit_results:
        doc = next((d for d in commits if d["id"] == did), None) if commits else None
        if not doc or doc["sha"] in seen_sha:
            continue
        seen_sha.add(doc["sha"])
        context_parts.append(format_commit(doc))
        if len(seen_sha) >= max_commits:
            break

    return "\n".join(context_parts)


def format_training_example(cards_context, commits_context, question, answer):
    """Format as MLX-compatible chat messages."""
    context_str = f"### Cards Context\n{cards_context}\n\n### Commits Context\n{commits_context}"
    user_content = (
        "Use the repo-arch cards and commit history as context. "
        "Answer based only on this evidence. Be specific about file paths, "
        "commit SHAs, package names, and numbers.\n\n"
        f"{context_str}\n\n"
        f"### Question\n{question}\n\n"
        "### Answer"
    )
    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]}


def generate_negative_questions():
    """Questions about topics not covered by repo history."""
    return [
        "What is the API rate limiting strategy?",
        "How does the CI/CD pipeline deploy to production?",
        "What monitoring and alerting systems are in place?",
        "How are feature flags managed?",
        "What is the database migration strategy?",
        "How is the project's documentation published?",
        "What is the security vulnerability disclosure process?",
        "How are A/B tests conducted?",
        "What is the backup and disaster recovery plan?",
        "How are third-party API integrations tested?",
        "What is the performance benchmarking methodology?",
        "How is the project's accessibility tested?",
        "What is the internationalization strategy?",
        "How are dependency upgrades scheduled?",
        "What is the rollback procedure for failed deployments?",
    ]


# ── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=800, help="Target example count")
    parser.add_argument("--out", default=".repo-arch/training-data/large")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--max-cards", type=int, default=3)
    parser.add_argument("--max-commits", type=int, default=6)
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo)
    out_dir = os.path.join(repo_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading indexes...", file=sys.stderr)
    cards, card_bm25, review = load_card_index(repo_dir)
    commits, commit_bm25 = load_commit_index(repo_dir)
    print(f"  Cards: {len(cards) if cards else 0}", file=sys.stderr)
    print(f"  Commits: {len(commits) if commits else 0}", file=sys.stderr)

    accepted_cards = [c for c in cards if c.get("status") == "accepted"] if cards else []
    print(f"  Accepted cards for generation: {len(accepted_cards)}", file=sys.stderr)

    examples = []
    total_needed = args.target
    neg_count = int(total_needed * args.negative_ratio)
    pos_count = total_needed - neg_count
    examples_per_card = max(2, pos_count // max(1, len(accepted_cards)))

    print(f"Target: {total_needed} examples ({pos_count} positive, {neg_count} negative)", file=sys.stderr)
    print(f"Examples per accepted card: {examples_per_card}", file=sys.stderr)

    # ── Generate positive examples ──
    gen_count = 0
    for card in accepted_cards:
        questions = generate_questions_for_card(card, count=examples_per_card + 4)

        for q in questions[:examples_per_card]:
            context_str = build_context(
                cards, card_bm25, commits, commit_bm25, q,
                max_cards=args.max_cards, max_commits=args.max_commits,
            )
            answer = generate_answer(context_str, q)
            example = format_training_example(
                "See context below", context_str, q, answer
            )
            example["source"] = f"card:{card.get('id','')[:12]}"
            examples.append(example)
            gen_count += 1

            if gen_count % 20 == 0:
                print(f"  Generated {gen_count}/{pos_count} positive examples...", file=sys.stderr)
            if gen_count >= pos_count:
                break
        if gen_count >= pos_count:
            break

    print(f"Generated {len(examples)} positive examples", file=sys.stderr)

    # ── Generate negative examples ──
    neg_questions = generate_negative_questions()
    random.shuffle(neg_questions)

    neg_gen = 0
    for q in neg_questions[:neg_count]:
        # For negatives, provide context that doesn't match
        context_str = build_context(
            cards, card_bm25, commits, commit_bm25, q,
            max_cards=2, max_commits=3,
        )
        answer = (
            "The available repo history does not contain information about this topic. "
            "The cards and commits cover code changes, co-change patterns, bug fixes, "
            "test coverage gaps, and reversion patterns within this repository. "
            "This question falls outside that scope."
        )
        example = format_training_example(
            "See context below", context_str, q, answer
        )
        example["source"] = "negative"
        examples.append(example)
        neg_gen += 1

    print(f"Generated {neg_gen} negative examples", file=sys.stderr)

    # ── Split and save ──
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.1))
    train_examples = examples[split:]
    valid_examples = examples[:split]

    def write_jsonl(path, exs):
        with open(path, "w") as f:
            for ex in exs:
                f.write(json.dumps({"messages": ex["messages"]}) + "\n")
        print(f"  {path}: {len(exs)} examples ({os.path.getsize(path)} bytes)", file=sys.stderr)

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train_examples)
    write_jsonl(os.path.join(out_dir, "valid.jsonl"), valid_examples)

    meta = {
        "total_examples": len(examples),
        "train": len(train_examples),
        "valid": len(valid_examples),
        "positive": len(examples) - neg_gen,
        "negative": neg_gen,
        "accepted_cards_used": len(accepted_cards),
        "examples_per_card": examples_per_card,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {json.dumps(meta, indent=2)}", file=sys.stderr)

    # Show sample
    if examples:
        print(f"\n=== Sample ===\n{json.dumps(examples[0], indent=2)[:1500]}", file=sys.stderr)

    print(f"\nDone. Dataset ready at {out_dir}")


if __name__ == "__main__":
    main()

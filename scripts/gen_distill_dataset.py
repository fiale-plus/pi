#!/usr/bin/env python3
"""
Self-distillation: generate training data matching the exact fused inference format.

For each synthetic question:
  1. Retrieve context (cards + commits) via BM25
  2. Generate answer with base model + fused context
  3. Filter low-quality answers
  4. Output: {context + question -> answer} in inference format

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/gen_distill_dataset.py --target 300 --out .repo-arch/training-data/distill
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
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── BM25 ──────────────────────────────────────────────────────────


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1; self.b = b
        self.documents = []; self.doc_id_to_idx = {}; self.idx_to_doc_id = {}
        self.avgdl = 0; self.N = 0; self.df = Counter(); self.term_in_docs = defaultdict(set)

    def tokenize(self, text):
        return re.findall(r'[a-zA-Z][a-zA-Z0-9_\-\./]{2,}', text.lower())

    def add_document(self, doc_id, text):
        idx = len(self.documents)
        self.doc_id_to_idx[doc_id] = idx; self.idx_to_doc_id[idx] = doc_id
        tokens = self.tokenize(text); self.documents.append(tokens)
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


def load_all(repo_dir):
    """Load cards + review + commit index."""
    # Cards
    cfiles = glob.glob(os.path.join(repo_dir, ".repo-arch", "cache", "cards", "*.json"))
    cards = []
    card_bm25 = BM25()
    if cfiles:
        with open(cfiles[0]) as f:
            data = json.load(f)
        cards = data.get("cards", [])
        rev_path = os.path.join(repo_dir, ".repo-arch", "review-state.json")
        review = json.load(open(rev_path)) if os.path.exists(rev_path) else {}
        for c in cards:
            cid = c.get("id", "")
            c["status"] = review.get(cid, {}).get("status", "unreviewed")
            text = f"{c.get('title','')} {c.get('suggestion','')} {' '.join(c.get('affectedFiles',[]))}"
            card_bm25.add_document(cid, text)

    # Commits
    idx_path = os.path.join(repo_dir, ".repo-arch", "index", "full", "index.json")
    commits = []
    commit_bm25 = BM25()
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            data = json.load(f)
        docs = data.get("documents", [])
        seen = set()
        for d in docs:
            if d.get("sha","") not in seen:
                seen.add(d["sha"]); commits.append(d)
        for doc in commits:
            commit_bm25.add_document(doc["id"], doc["text"])

    return cards, card_bm25, commits, commit_bm25


# ── Format context (exact match of fused_answer.py) ──────────────


def format_card(card):
    affected = card.get("affectedFiles", [])
    suggestion = card.get("suggestion", "")[:300]
    return f'<CARD type="{card.get("type","")}" status="{card.get("status","")}" confidence="{card.get("confidence",0)}">\n' \
           f'  Files: {", ".join(affected[:3])}\n' \
           f'  Summary: {suggestion}\n' \
           f'</CARD>'


def format_commit(doc):
    pkgs = doc.get("packages", [])
    files = doc.get("paths", [])[:4]
    lines = [
        f'<COMMIT sha="{doc.get("sha","")[:12]}">',
        f'  Subject: {doc.get("subject","")[:200]}',
    ]
    if pkgs: lines.append(f'  Packages: {", ".join(pkgs[:3])}')
    if files: lines.append(f'  Files: {", ".join(files)}')
    lines.append('</COMMIT>')
    return "\n".join(lines)


def build_fused_context(question, cards, card_bm25, commits, commit_bm25, max_cards=4, max_commits=6):
    """Build context matching fused_answer.py format exactly."""
    parts = ["--- REPO-ARCH CARDS ---"]

    card_results = card_bm25.search(question, top_k=max_cards * 2) if card_bm25 else []
    seen_types = set()
    added = 0
    for cid, score in card_results:
        c = next((c for c in cards if c["id"] == cid), None) if cards else None
        if not c or c.get("status") == "rejected" or c.get("type","") in seen_types:
            continue
        seen_types.add(c.get("type",""))
        parts.append(format_card(c))
        added += 1
        if added >= max_cards:
            break

    parts.append("\n--- COMMIT HISTORY ---")
    commit_results = commit_bm25.search(question, top_k=max_commits * 2) if commit_bm25 else []
    seen_sha = set()
    for did, score in commit_results:
        doc = next((d for d in commits if d["id"] == did), None) if commits else None
        if not doc or doc["sha"] in seen_sha:
            continue
        seen_sha.add(doc["sha"])
        parts.append(format_commit(doc))
        if len(seen_sha) >= max_commits:
            break

    return "\n".join(parts)


SYSTEM_PROMPT = (
    "You are a repo-aware coding assistant for the pi monorepo. "
    "Answer using only the repo-arch cards and commit history provided as context. "
    "Be specific about file paths, commit SHAs, package names, and commit counts. "
    "If the context does not contain the answer, say what information is missing."
)


def generate_answer(context_str, question, max_tokens=384, temperature=0.3):
    """Generate teacher answer using base model + fused context."""
    user_prompt = f"### Context\n{context_str}\n\n### Question\n{question}\n\n### Answer"
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
        "--system-prompt", SYSTEM_PROMPT,
        "--prompt", user_prompt,
        "--use-default-chat-template",
    ]
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start
        output = result.stdout.strip()
        parts = output.split("==========")
        answer = parts[-2].strip() if len(parts) >= 3 and parts[-2].strip() else parts[-3].strip() if len(parts) >= 3 else output
        for line in answer.split("\n"):
            if "tokens-per-sec" in line or "Peak memory" in line or "Prompt:" in line:
                answer = answer.replace(line, "")
        answer = answer.strip()
        return {"answer": answer, "latency": round(elapsed, 2), "error": None}
    except Exception as e:
        return {"answer": None, "latency": 0, "error": str(e)}


def format_example(context_str, question, answer):
    """Format as MLX-compatible chat training example, matching inference format."""
    user_content = (
        "Use the repo-arch cards and commit history as context. "
        "Answer based only on this evidence. Be specific about file paths, "
        "commit SHAs, package names, and commit counts.\n\n"
        f"{context_str}\n\n"
        f"### Question\n{question}\n\n"
        "### Answer"
    )
    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]}


def quality_filter(result):
    """Filter low-quality teacher answers."""
    if result.get("error"):
        return False
    answer = result.get("answer", "")
    if not answer or len(answer) < 20:
        return False
    # Remove empty/vague answers
    vague = ["i do not know", "i don't know", "i am not sure", "i cannot answer",
             "no information", "not enough information", "the context does not contain",
             "based on the provided context", "based on the context"]
    if any(v in answer.lower()[:80] for v in vague):
        return False
    # Must have at least one specific reference
    has_ref = bool(re.search(r'packages/[\w/-]+|[a-f0-9]{7,12}|\d+\s+(?:times|commits|fixes)', answer.lower()))
    return has_ref


# ── Question generation ──────────────────────────────────────────


def generate_questions(cards, commits):
    """Generate diverse synthetic questions from cards and commit clusters."""
    questions = set()

    # From accepted cards
    accepted = [c for c in cards if c.get("status") == "accepted"] if cards else []
    for card in accepted:
        affected = card.get("affectedFiles", [])
        ctype = card.get("type", "")
        for f in affected[:2]:
            questions.add(f"What should I know before modifying {f}?")
            questions.add(f"What are the risks of changing {f}?")
            if ctype == "repeated-fix":
                questions.add(f"How many times has {f} been fixed?")
                questions.add(f"What keeps breaking in {f}?")
            elif ctype == "test-gap":
                questions.add(f"Is there test coverage for {f}?")
                questions.add(f"What changed in {f} without tests?")
            elif ctype == "co-change":
                questions.add(f"What files co-change with {f}?")
            elif ctype == "revert-pattern":
                questions.add(f"Has {f} been unstable?")
            elif ctype == "high-churn":
                questions.add(f"How often does {f} change?")

    # From top commit clusters
    if commits:
        # Group commits by file
        file_clusters = defaultdict(list)
        for doc in commits[:2000]:
            for p in doc.get("paths", []):
                if "CHANGELOG" not in p and "/index." not in p:
                    file_clusters[p].append(doc)

        top_files = sorted(file_clusters.keys(), key=lambda f: -len(file_clusters[f]))[:80]
        for f in top_files:
            clusters = file_clusters[f]
            questions.add(f"How many commits touched {f}?")
            if len(clusters) > 5:
                questions.add(f"What is the commit history of {f}?")
                subjects = [c.get("subject","") for c in clusters]
                if any("fix" in s.lower() for s in subjects):
                    questions.add(f"What bugs were fixed in {f}?")
                if any("test" in s.lower() for s in subjects):
                    questions.add(f"What test changes affected {f}?")
                if any("refactor" in s.lower() for s in subjects):
                    questions.add(f"What refactors affected {f}?")

    # Cross-cutting questions
    cross_cutting = [
        "Which packages have the most bug fix commits?",
        "Which packages have test coverage gaps?",
        "What files co-change most frequently across packages?",
        "Which files are most risky to modify?",
        "What is the most frequently changed file in the coding-agent package?",
        "What is the most frequently changed file in the ai package?",
        "What is the most frequently changed file in the tui package?",
        "How do model updates typically cascade through packages?",
        "What patterns exist in the reversion history?",
        "Which config files have high churn rates?",
    ]
    for q in cross_cutting:
        questions.add(q)

    return list(questions)


# ── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=300)
    parser.add_argument("--out", default=".repo-arch/training-data/distill")
    parser.add_argument("--max-cards", type=int, default=4)
    parser.add_argument("--max-commits", type=int, default=6)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.join(REPO_DIR, args.out)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading indexes...", file=sys.stderr)
    cards, card_bm25, commits, commit_bm25 = load_all(REPO_DIR)
    print(f"  Cards: {len(cards)}, Commits: {len(commits)}", file=sys.stderr)

    if not args.skip_generation:
        # Generate and deduplicate questions
        all_questions = generate_questions(cards, commits)
        random.shuffle(all_questions)
        selected = all_questions[:args.target]
        # Save questions for reproducibility
        with open(os.path.join(out_dir, "questions.json"), "w") as f:
            json.dump(selected, f, indent=2)
        print(f"Generated {len(selected)} questions (from {len(all_questions)} candidates)", file=sys.stderr)
    else:
        with open(os.path.join(out_dir, "questions.json")) as f:
            selected = json.load(f)
        print(f"Loaded {len(selected)} questions from cache", file=sys.stderr)

    # Generate training examples
    examples = []
    skipped = 0
    total = len(selected)

    for i, question in enumerate(selected):
        print(f"[{i+1}/{total}] ... ", end="", file=sys.stderr, flush=True)

        context = build_fused_context(
            question, cards, card_bm25, commits, commit_bm25,
            max_cards=args.max_cards, max_commits=args.max_commits,
        )
        result = generate_answer(context, question)

        if quality_filter(result):
            example = format_example(context, question, result["answer"])
            example["source_question"] = question[:100]
            examples.append(example)
            print(f"ok ({result['latency']}s)", file=sys.stderr)
        else:
            skipped += 1
            print(f"skip ({result.get('error','low quality')})", file=sys.stderr)

    print(f"\nGenerated {len(examples)} examples, skipped {skipped}", file=sys.stderr)

    # Split
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.1))
    train = examples[split:]
    valid = examples[:split]

    def write_jsonl(path, exs):
        with open(path, "w") as f:
            for ex in exs:
                f.write(json.dumps({"messages": ex["messages"]}) + "\n")
        print(f"  {path}: {len(exs)} examples ({os.path.getsize(path)} bytes)", file=sys.stderr)

    write_jsonl(os.path.join(out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(out_dir, "valid.jsonl"), valid)

    meta = {
        "total_generated": len(examples),
        "skipped": skipped,
        "train": len(train),
        "valid": len(valid),
        "questions_candidates": len(selected),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {json.dumps(meta, indent=2)}", file=sys.stderr)

    if examples:
        print(f"\n=== Sample ===\n{json.dumps(examples[0], indent=2)[:1200]}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fused retrieval: combine repo-arch card context + full commit history context.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/fused_answer.py "What should I know about agent-session.ts?"
  python3 scripts/fused_answer.py --eval .repo-arch/eval/questions.jsonl --out results.jsonl
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

random = __import__("random")

# ── BM25 (same as full_history_answer) ───────────────────────────


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_id_to_idx = {}
        self.idx_to_doc_id = {}
        self.avgdl = 0
        self.N = 0
        self.df = Counter()
        self.term_in_docs = defaultdict(set)

    def tokenize(self, text):
        return re.findall(r'[a-zA-Z][a-zA-Z0-9_\-\./]{2,}', text.lower())

    def add_document(self, doc_id, text):
        idx = len(self.documents)
        self.doc_id_to_idx[doc_id] = idx
        self.idx_to_doc_id[idx] = doc_id
        tokens = self.tokenize(text)
        self.documents.append(tokens)
        for term in set(tokens):
            self.df[term] += 1
            self.term_in_docs[term].add(idx)
        self.N = len(self.documents)
        self.avgdl = sum(len(d) for d in self.documents) / max(1, self.N)

    def idf(self, term):
        n = self.df.get(term, 0)
        return 0 if n == 0 else math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query, top_k=20):
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = defaultdict(float)
        for term in tokens:
            idf_val = self.idf(term)
            if idf_val == 0:
                continue
            for doc_idx in self.term_in_docs.get(term, set()):
                dl = len(self.documents[doc_idx])
                tf = self.documents[doc_idx].count(term)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_idx] += idf_val * tf * (self.k1 + 1) / max(denom, 0.001)
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return [(self.idx_to_doc_id.get(doc_idx, ""), score) for doc_idx, score in sorted_scores[:top_k]]


# ── Load all indexes ─────────────────────────────────────────────


def load_card_index(repo_dir):
    """Load repo-arch cards."""
    cfiles = glob.glob(os.path.join(repo_dir, ".repo-arch", "cache", "cards", "*.json"))
    if not cfiles:
        return None, None
    with open(cfiles[0]) as f:
        data = json.load(f)
    cards = data.get("cards", [])

    review_path = os.path.join(repo_dir, ".repo-arch", "review-state.json")
    if os.path.exists(review_path):
        with open(review_path) as f:
            review = json.load(f)
    else:
        review = {}

    # Index cards in BM25
    bm25 = BM25()
    for card in cards:
        cid = card.get("id", "")
        status = review.get(cid, {}).get("status", "unreviewed")
        card["status"] = status
        if status != "rejected":
            text = f"{card.get('title','')} {card.get('suggestion','')} {' '.join(card.get('affectedFiles',[]))}"
            bm25.add_document(cid, text)
    return cards, bm25


def load_commit_index(repo_dir):
    """Load full commit history index."""
    index_path = os.path.join(repo_dir, ".repo-arch", "index", "full", "index.json")
    if not os.path.exists(index_path):
        return None, None
    with open(index_path) as f:
        data = json.load(f)
    docs = data.get("documents", [])
    # Dedup
    seen = set()
    unique = []
    for d in docs:
        sha = d.get("sha", "")
        if sha not in seen:
            seen.add(sha)
            unique.append(d)
    docs = unique
    bm25 = BM25()
    for doc in docs:
        bm25.add_document(doc["id"], doc["text"])
    return docs, bm25


# ── Fused context ────────────────────────────────────────────────


def format_card_text(card):
    """Compact single-line card representation."""
    ctype = card.get("type", "unknown")
    affected = card.get("affectedFiles", [])
    suggestion = card.get("suggestion", "")[:200]
    files = ", ".join(affected[:3])
    return f"[CARD: {ctype}] {files} | {suggestion}"


def format_commit_text(doc):
    """Compact single-line commit representation."""
    pkgs = doc.get("packages", [])
    files = doc.get("paths", [])
    pkg_str = f" [{', '.join(pkgs[:2])}]" if pkgs else ""
    file_str = ""
    if files:
        file_str = " files: " + ", ".join(files[:3])
    return f"[{doc['sha']}]{pkg_str}{file_str} {doc.get('subject','')[:200]}"


# ── Answer ────────────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are a repo-aware coding assistant for the pi monorepo. "
    "You are given two types of context:\n"
    "1. REPO-ARCH CARDS: Pattern-level summaries (co-change clusters, repeated fixes, test gaps)\n"
    "2. COMMIT HISTORY: Individual commits with file paths\n\n"
    "Answer using both. Be specific about file paths, commit SHAs, package names, and commit counts. "
    "If context is insufficient, say what information is missing."
)


def answer(context, question, max_tokens=512, temperature=0.3):
    user_prompt = f"### Context\n{context}\n\n### Question\n{question}"
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start
        output = result.stdout.strip()
        parts = output.split("==========")
        answer = parts[-2].strip() if len(parts) >= 3 and parts[-2].strip() else parts[-3].strip() if len(parts) >= 3 else output
        for line in answer.split("\n"):
            if "tokens-per-sec" in line or "Peak memory" in line:
                answer = answer.replace(line, "")
        return {"answer": answer.strip(), "latency": round(elapsed, 2), "error": None}
    except subprocess.TimeoutExpired:
        return {"answer": None, "latency": 120, "error": "timeout"}
    except Exception as e:
        return {"answer": None, "latency": 0, "error": str(e)}


# ── Main ──────────────────────────────────────────────────────────


def build_fused_context(question, cards, card_bm25, commits, commit_bm25, max_cards=5, max_commits=8):
    """Fuse card + commit results into a single context block."""
    # Cards
    card_results = card_bm25.search(question, top_k=max_cards * 2) if card_bm25 else []
    context_lines = ["--- REPO-ARCH CARDS (pattern summaries) ---"]
    seen = set()
    for cid, score in card_results:
        card = next((c for c in cards if c["id"] == cid), None) if cards else None
        if not card or card.get("status") == "rejected":
            continue
        text = format_card_text(card)
        if text[:80] not in seen:
            seen.add(text[:80])
            context_lines.append(text)
        if len(context_lines) - 1 >= max_cards:
            break

    # Commits
    commit_results = commit_bm25.search(question, top_k=max_commits * 2) if commit_bm25 else []
    context_lines.append("\n--- COMMIT HISTORY (individual commits) ---")
    seen_sha = set()
    for did, score in commit_results:
        doc = next((d for d in commits if d["id"] == did), None) if commits else None
        if not doc or doc["sha"] in seen_sha:
            continue
        seen_sha.add(doc["sha"])
        context_lines.append(format_commit_text(doc))
        if len(seen_sha) >= max_commits:
            break

    return "\n".join(context_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--question", "-q", dest="qflag")
    parser.add_argument("--eval")
    parser.add_argument("--out", default=".repo-arch/eval/runs/fused-45.jsonl")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-cards", type=int, default=5)
    parser.add_argument("--max-commits", type=int, default=8)
    args = parser.parse_args()

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print("Loading indexes...", file=sys.stderr)
    cards, card_bm25 = load_card_index(repo_dir)
    commits, commit_bm25 = load_commit_index(repo_dir)
    print(f"  Cards: {len(cards) if cards else 0}", file=sys.stderr)
    print(f"  Commits: {len(commits) if commits else 0}", file=sys.stderr)

    # ── Eval mode ──
    if args.eval:
        questions_path = os.path.join(repo_dir, args.eval)
        with open(questions_path) as f:
            questions = [json.loads(l) for l in f]
        out_path = os.path.join(repo_dir, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w") as out_f:
            for i, q in enumerate(questions):
                qid, question = q["id"], q["question"]
                print(f"[{i+1}/{len(questions)}] Q{qid} ... ", end="", file=sys.stderr, flush=True)

                context = build_fused_context(
                    question, cards, card_bm25, commits, commit_bm25,
                    max_cards=args.max_cards, max_commits=args.max_commits,
                )
                result = answer(context, question)
                rec = {
                    "question_id": qid,
                    "question": question,
                    "mode": "fused",
                    "answer": result.get("answer"),
                    "latency": result.get("latency"),
                    "error": result.get("error"),
                }
                out_f.write(json.dumps(rec) + "\n")
                print("ok" if not result.get("error") else "ERR", file=sys.stderr)

        print(f"\nDone. Results: {out_path}", file=sys.stderr)
        return

    # ── Single question ──
    question = args.question or args.qflag
    if not question:
        parser.print_help()
        sys.exit(1)

    context = build_fused_context(
        question, cards, card_bm25, commits, commit_bm25,
        max_cards=args.max_cards, max_commits=args.max_commits,
    )
    result = answer(context, question)

    if args.json:
        print(json.dumps({
            "question": question,
            "answer": result.get("answer"),
            "latency": result.get("latency"),
        }, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  Question: {question}")
        print(f"{'='*60}")
        print(f"\n  Answer ({result['latency']}s):")
        print(f"  {result.get('answer', 'ERROR: ' + str(result.get('error')))}")


if __name__ == "__main__":
    main()

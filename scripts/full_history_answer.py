#!/usr/bin/env python3
"""
Full-history answer: retrieve from all 4089 commits + answer with base model.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/full_history_answer.py "What should I know about agent-session.ts?"
  python3 scripts/full_history_answer.py --question "..." --top-k 10 --json
  python3 scripts/full_history_answer.py --eval .repo-arch/eval/questions.jsonl --out results.jsonl
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict


# ── BM25 ──────────────────────────────────────────────────────────


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
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        scores = defaultdict(float)
        for term in query_tokens:
            idf_val = self.idf(term)
            if idf_val == 0:
                continue
            for doc_idx in self.term_in_docs.get(term, set()):
                doc_len = len(self.documents[doc_idx])
                tf = self.documents[doc_idx].count(term)
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_idx] += idf_val * tf * (self.k1 + 1) / max(denom, 0.001)
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return [(self.idx_to_doc_id.get(doc_idx, ""), score)
                for doc_idx, score in sorted_scores[:top_k]]


# ── Load ──────────────────────────────────────────────────────────


def load_index(index_dir):
    path = os.path.join(index_dir, "index.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
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


# ── Format context ────────────────────────────────────────────────


def format_context(docs, results, max_commits=10):
    """Format top unique results into compact context."""
    seen_texts = set()
    parts = []
    for doc_id, score in results:
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if not doc:
            continue
        subj = doc.get("subject", "")
        if subj[:80] in seen_texts:
            continue
        seen_texts.add(subj[:80])

        pkgs = doc.get("packages", [])
        pkg_str = f" [{', '.join(pkgs[:3])}]" if pkgs else ""
        files = doc.get("paths", [])
        file_str = ""
        if files:
            # Show files that match packages in the files list
            file_str = " files: " + ", ".join(files[:4])

        parts.append(
            f"[{score:.2f}] {doc['sha']}: {subj[:200]}{pkg_str}{file_str}"
        )
        if len(parts) >= max_commits:
            break

    return "\n".join(parts)


SYSTEM_PROMPT = (
    "You are a repo-aware coding assistant. "
    "Answer using only the commit history provided as context. "
    "Be specific about file paths, commit counts, package names, and commit SHAs. "
    "If the context doesn't contain enough information, say what is missing."
)


def answer_with_model(context, question, max_tokens=512, temperature=0.3):
    """Answer with base Qwen model."""
    user_prompt = (
        "Here is relevant commit history from the repository:\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the commit history above."
    )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Question")
    parser.add_argument("--question", "-q", dest="qflag")
    parser.add_argument("--index", default=".repo-arch/index/full")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--eval", help="Run on eval questions JSONL")
    parser.add_argument("--out", default=".repo-arch/eval/runs/full-history.jsonl")
    args = parser.parse_args()

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_dir = os.path.join(repo_dir, args.index)

    docs, bm25 = load_index(index_dir)
    if docs is None:
        print("No index. Run scripts/build_index.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(docs)} unique commits", file=sys.stderr)

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

                results = bm25.search(question, top_k=args.top_k)
                context = format_context(docs, results, max_commits=args.top_k)
                result = answer_with_model(context, question)

                rec = {
                    "question_id": qid,
                    "question": question,
                    "mode": "full-history",
                    "answer": result.get("answer"),
                    "latency": result.get("latency"),
                    "num_commits_retrieved": len(results),
                    "error": result.get("error"),
                }
                out_f.write(json.dumps(rec) + "\n")
                status = "ok" if not result.get("error") else f"ERR"
                print(status, file=sys.stderr)

        print(f"\nResults: {out_path}", file=sys.stderr)
        return

    # ── Single question ──
    question = args.question or args.qflag
    if not question:
        parser.print_help()
        sys.exit(1)

    results = bm25.search(question, top_k=args.top_k)
    context = format_context(docs, results, max_commits=args.top_k)
    result = answer_with_model(context, question)

    if args.json:
        print(json.dumps({
            "question": question,
            "context_commits": len(results),
            "answer": result.get("answer"),
            "latency": result.get("latency"),
        }, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"  Question: {question}")
        print(f"  Context: {len(results)} commits")
        print(f"{'='*60}")
        print(f"\n  Answer ({result['latency']}s):")
        print(f"  {result.get('answer', 'ERROR: ' + str(result.get('error')))}")


if __name__ == "__main__":
    main()

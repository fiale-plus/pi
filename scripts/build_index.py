#!/usr/bin/env python3
"""
Full-history indexer: build a searchable index from ALL repo-arch mined history.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/build_index.py --repo . --history .repo-arch/cache/history-*.jsonl
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ── BM25 implementation (pure Python, no deps) ───────────────────


class BM25:
    """Simple BM25 index without external dependencies."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_id_to_idx = {}
        self.idx_to_doc_id = {}
        self.avgdl = 0
        self.N = 0
        self.df = Counter()  # term -> doc frequency
        self.term_in_docs = defaultdict(set)

    def tokenize(self, text):
        """Simple tokenizer: lowercase, split on non-alphanumeric, filter short tokens."""
        tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9_\-\./]{1,}', text.lower())
        return tokens

    def add_document(self, doc_id, text):
        """Add a document to the index."""
        idx = len(self.documents)
        self.doc_id_to_idx[doc_id] = idx
        self.idx_to_doc_id[idx] = doc_id

        tokens = self.tokenize(text)
        self.documents.append(tokens)

        # Update term frequencies
        unique_terms = set(tokens)
        for term in unique_terms:
            self.df[term] += 1
            self.term_in_docs[term].add(idx)

        self.N = len(self.documents)
        self.avgdl = sum(len(d) for d in self.documents) / max(1, self.N)

    def idf(self, term):
        """Inverse document frequency."""
        n = self.df.get(term, 0)
        if n == 0:
            return 0
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query, top_k=20):
        """Search BM25 index, return list of (doc_id, score)."""
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
                # BM25 scoring
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_idx] += idf_val * numerator / denominator

        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        results = []
        for doc_idx, score in sorted_scores[:top_k]:
            doc_id = self.idx_to_doc_id[doc_idx]
            results.append((doc_id, score))

        return results


# ── Commit document builder ──────────────────────────────────────


def build_commit_document(commit):
    """Build a searchable document from a commit record."""
    sha = commit.get("sha", "")
    subject = commit.get("subject", "")
    files = commit.get("files", [])
    paths = commit.get("paths", [])
    signals = commit.get("signals", [])
    author = commit.get("author", {})
    date = commit.get("authoredAt", "")

    # Extract file paths and statuses
    file_paths = [f.get("path", "") for f in files]
    file_statuses = {f.get("path", ""): f.get("status", "M") for f in files}

    # Extract packages from paths
    packages = set()
    for p in paths:
        m = re.match(r'^(packages/[\w-]+)', p)
        if m:
            packages.add(m.group(1))

    # Extract signal types
    signal_types = [s.get("type", "") for s in signals]
    signal_labels = [s.get("label", "") for s in signals]

    # Build text for search (weighted fields)
    title_text = subject.lower()
    path_text = " ".join(p.lower() for p in paths)
    package_text = " ".join(sorted(packages)).lower()
    signal_text = " ".join(signal_labels + signal_types).lower()

    # Full text for indexing
    full_text = f"{title_text} {path_text} {package_text} {signal_text}"

    return {
        "id": sha[:16],
        "sha": sha[:12],
        "subject": subject,
        "files": file_paths[:20],  # limit
        "paths": paths,
        "packages": sorted(packages),
        "signals": signal_types,
        "date": date,
        "author": author.get("name", ""),
        "text": full_text,
        "file_statuses": file_statuses,
    }


def load_history(history_glob):
    """Load all commits from history JSONL files."""
    files = glob.glob(history_glob)
    if not files:
        # Try bare path
        files = [history_glob]
    commits = []
    for fpath in files:
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    commits.append(json.loads(line))
    return commits


def main():
    parser = argparse.ArgumentParser(description="Build full-history search index")
    parser.add_argument("--repo", default=".", help="Repo path")
    parser.add_argument("--history", default=".repo-arch/cache/history-*.jsonl",
                        help="History JSONL glob")
    parser.add_argument("--out", default=".repo-arch/index/full", help="Output dir")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of results to test")
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo)
    history_path = os.path.join(repo_dir, args.history)
    out_dir = os.path.join(repo_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading history from {history_path}...")
    commits = load_history(history_path)
    print(f"Loaded {len(commits)} commits")

    # Build documents
    print("Building commit documents...")
    docs = []
    bm25 = BM25()
    for commit in commits:
        doc = build_commit_document(commit)
        doc_id = doc["id"]
        bm25.add_document(doc_id, doc["text"])
        docs.append(doc)

    # Save index
    index_data = {
        "num_commits": len(docs),
        "documents": docs,
    }
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"Saved index: {index_path} ({len(docs)} docs, {os.path.getsize(index_path)} bytes)")

    # ── Test retrieval on all 45 eval questions ──
    questions_path = os.path.join(repo_dir, ".repo-arch", "eval", "questions.jsonl")
    if os.path.exists(questions_path):
        print(f"\nTesting retrieval on 45 eval questions...")
        with open(questions_path) as f:
            questions = [json.loads(l) for l in f]

        # Count how many questions get at least one relevant result
        # "Relevant" = result contains a package path from the question
        relevant_found = 0
        total_pkg_refs = 0

        for q in questions:
            qid = q["id"]
            question = q["question"]
            # Extract packages from question
            q_packages = set(re.findall(r'packages/[\w/-]+', question))

            results = bm25.search(question, top_k=args.top_k)
            found_relevant = False
            pkg_refs = 0

            for doc_id, score in results:
                # Find the matching doc
                doc = next((d for d in docs if d["id"] == doc_id), None)
                if not doc:
                    continue
                # Count package refs in results
                for p in doc.get("packages", []):
                    total_pkg_refs += 1
                    if p in question.lower():
                        found_relevant = True
                # Also check if file paths match
                for f in doc.get("paths", []):
                    if any(qp in f for qp in q_packages):
                        found_relevant = True

            if found_relevant:
                relevant_found += 1

        recall = (relevant_found / len(questions)) * 100
        print(f"  Questions with relevant context: {relevant_found}/{len(questions)} ({recall:.0f}%)")
        print(f"  Total package refs in top-{args.top_k}: {total_pkg_refs}")
        print(f"  Avg package refs per question: {total_pkg_refs / len(questions):.1f}")

        # Show questions with poor retrieval
        print(f"\nQuestions with <5 package refs in top-{args.top_k}:")
        for q in questions:
            qid = q["id"]
            question = q["question"]
            q_packages = set(re.findall(r'packages/[\w/-]+', question))
            results = bm25.search(question, top_k=args.top_k)
            pkg_refs = 0
            for doc_id, score in results:
                doc = next((d for d in docs if d["id"] == doc_id), None)
                if doc:
                    for p in doc.get("packages", []):
                        if any(qp in p for qp in q_packages):
                            pkg_refs += 1
            if pkg_refs < 5:
                print(f"  Q{qid}: {question[:70]} ({pkg_refs} pkg refs)")

        # Also check for the questions that had ZERO useful context in the original retrieval
        print(f"\nWorst original questions (config, TUI, web-ui):")
        orig_worst = [10, 13, 14, 17, 18]
        for qid in orig_worst:
            q = next((qq for qq in questions if qq["id"] == qid), None)
            if not q:
                continue
            results = bm25.search(q["question"], top_k=args.top_k)
            top5 = []
            for doc_id, score in results[:5]:
                doc = next((d for d in docs if d["id"] == doc_id), None)
                if doc:
                    top5.append(f"{doc['sha']}: {doc['subject'][:60]}")
            print(f"  Q{qid}: {q['question'][:70]}")
            for t in top5:
                print(f"    {t}")

    print(f"\nDone. Use scripts/full_retrieve.py to search the index.")


if __name__ == "__main__":
    main()

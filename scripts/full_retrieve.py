#!/usr/bin/env python3
"""
Full-history retrieval: search the commit index for relevant context.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/full_retrieve.py "What should I know about agent-session.ts?" --top-k 10
  python3 scripts/full_retrieve.py "config changes without tests" --top-k 5 --json
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict


# ── BM25 (same implementation as build_index) ────────────────────


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
        tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9_\-\./]{1,}', text.lower())
        return tokens

    def add_document(self, doc_id, text):
        idx = len(self.documents)
        self.doc_id_to_idx[doc_id] = idx
        self.idx_to_doc_id[idx] = doc_id
        tokens = self.tokenize(text)
        self.documents.append(tokens)
        unique_terms = set(tokens)
        for term in unique_terms:
            self.df[term] += 1
            self.term_in_docs[term].add(idx)
        self.N = len(self.documents)
        self.avgdl = sum(len(d) for d in self.documents) / max(1, self.N)

    def idf(self, term):
        n = self.df.get(term, 0)
        if n == 0:
            return 0
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)

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
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_idx] += idf_val * numerator / denominator
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return [(self.idx_to_doc_id[doc_idx], score) for doc_idx, score in sorted_scores[:top_k]]


# ── Search ────────────────────────────────────────────────────────


def load_index(index_dir):
    """Load the full-history index."""
    index_path = os.path.join(index_dir, "index.json")
    if not os.path.exists(index_path):
        return None, None

    with open(index_path) as f:
        data = json.load(f)

    docs = data.get("documents", [])
    # Deduplicate by SHA
    seen_shas = set()
    unique_docs = []
    for doc in docs:
        sha = doc.get("sha", "")
        if sha not in seen_shas:
            seen_shas.add(sha)
            unique_docs.append(doc)

    print(f"Deduplicated: {len(docs)} -> {len(unique_docs)} unique commits")
    docs = unique_docs

    # Rebuild BM25 from deduplicated doc texts
    bm25 = BM25()
    for doc in docs:
        bm25.add_document(doc["id"], doc["text"])
    return docs, bm25


def format_context(docs, results, max_cards=8):
    """Format top results into a context block for the LLM."""
    seen_subjects = set()
    context_parts = []

    for doc_id, score in results[:max_cards]:
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if not doc:
            continue

        # Deduplicate similar subjects
        subject_key = doc.get("subject", "")[:80]
        if subject_key in seen_subjects:
            continue
        seen_subjects.add(subject_key)

        lines = [f'<COMMIT sha="{doc["sha"]}" score="{score:.3f}">']
        lines.append(f"Subject: {doc.get('subject', '')[:200]}")
        lines.append(f"Files ({len(doc.get('paths', []))}):")
        for p in doc.get("paths", [])[:8]:
            lines.append(f"  {p}")
        if doc.get("packages"):
            lines.append(f"Packages: {', '.join(doc['packages'][:5])}")
        lines.append("</COMMIT>")
        context_parts.append("\n".join(lines))

    return "\n\n".join(context_parts)


def main():
    parser = argparse.ArgumentParser(description="Full-history retrieval")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--query", "-q", dest="qflag", help="Alternative query flag")
    parser.add_argument("--index", default=".repo-arch/index/full", help="Index directory")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--context", action="store_true", help="Output formatted context")

    args = parser.parse_args()
    query = args.query or args.qflag
    if not query:
        parser.print_help()
        sys.exit(1)

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_dir = os.path.join(repo_dir, args.index)

    docs, bm25 = load_index(index_dir)
    if docs is None:
        print("No index found. Run scripts/build_index.py first.", file=sys.stderr)
        sys.exit(1)

    results = bm25.search(query, top_k=args.top_k)

    if args.json:
        output = []
        for doc_id, score in results:
            doc = next((d for d in docs if d["id"] == doc_id), None)
            if doc:
                output.append({
                    "sha": doc["sha"],
                    "score": round(score, 3),
                    "subject": doc.get("subject", "")[:150],
                    "packages": doc.get("packages", []),
                    "files": doc.get("paths", [])[:5],
                })
        print(json.dumps({"query": query, "results": output}, indent=2))
    elif args.context:
        context = format_context(docs, results, max_cards=args.top_k)
        print(context)
    else:
        print(f"\n{'='*60}")
        print(f"  Query: {query}")
        print(f"{'='*60}")
        for doc_id, score in results:
            doc = next((d for d in docs if d["id"] == doc_id), None)
            if doc:
                pkgs = ", ".join(doc.get("packages", [])[:3])
                print(f"\n  [{score:.3f}] {doc['sha']}: {doc.get('subject', '')[:100]}")
                print(f"    Files: {', '.join(doc.get('paths', [])[:4])}")
                if pkgs:
                    print(f"    Packages: {pkgs}")


if __name__ == "__main__":
    main()

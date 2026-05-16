#!/usr/bin/env python3
"""
Build hybrid search index: BM25 + embeddings for all 4089 commits.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/build_hybrid_index.py --repo .
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── BM25 ─────────────────────────────────────────────────────────


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1=k1;self.b=b;self.doc=[];self.d2i={};self.i2d={};self.avgdl=0;self.N=0;self.df=Counter();self.tid=defaultdict(set)
    def tok(self,t): return re.findall(r'[a-zA-Z][a-zA-Z0-9_\-\./]{2,}',t.lower())
    def add(self,id_,text):
        i=len(self.doc);self.d2i[id_]=i;self.i2d[i]=id_
        t=self.tok(text);self.doc.append(t)
        for x in set(t): self.df[x]+=1;self.tid[x].add(i)
        self.N=len(self.doc);self.avgdl=sum(len(d)for d in self.doc)/max(1,self.N)
    def idf(self,t): n=self.df.get(t,0);return 0 if n==0 else math.log((self.N-n+0.5)/(n+0.5)+1)
    def search(self,q,k=20):
        t=self.tok(q);s=defaultdict(float)
        for x in t:
            i=self.idf(x)
            if i==0: continue
            for d in self.tid.get(x,set()):
                dl=len(self.doc[d]);tf=self.doc[d].count(x)
                denom=tf+self.k1*(1-self.b+self.b*dl/self.avgdl)
                s[d]+=i*tf*(self.k1+1)/max(denom,0.001)
        return [(self.i2d.get(i,''),sc) for i,sc in sorted(s.items(),key=lambda x:-x[1])[:k]]
    def search_all_scores(self,q,k=200):
        """Return all scores for RRF fusion."""
        t=self.tok(q);s=defaultdict(float)
        for x in t:
            i=self.idf(x)
            if i==0: continue
            for d in self.tid.get(x,set()):
                dl=len(self.doc[d]);tf=self.doc[d].count(x)
                denom=tf+self.k1*(1-self.b+self.b*dl/self.avgdl)
                s[d]+=i*tf*(self.k1+1)/max(denom,0.001)
        return {self.i2d.get(i,''): sc for i,sc in s.items()}


# ── Embedding ────────────────────────────────────────────────────


def build_embeddings(commits, model_name="all-MiniLM-L6-v2"):
    """Build embeddings for all commits using sentence-transformers or similar."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...", file=sys.stderr)
        import subprocess as sp
        sp.run([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"], check=True)
        from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}...", file=sys.stderr)
    model = SentenceTransformer(model_name)
    model.to("cpu")  # ensure CPU, memory-safe

    # Build texts to embed
    texts = []
    doc_map = {}  # doc_id -> index
    for i, doc in enumerate(commits):
        sha = doc.get("sha", "")[:12]
        subject = doc.get("subject", "")
        paths = " ".join(doc.get("paths", [])[:5])
        packages = " ".join(doc.get("packages", []))
        text = f"{subject} {paths} {packages}"
        texts.append(text)
        doc_map[doc["id"]] = i

    print(f"Embedding {len(texts)} documents...", file=sys.stderr)
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    elapsed = time.time() - start
    print(f"Embedded {len(embeddings)} docs in {elapsed:.1f}s ({len(texts)/elapsed:.0f} docs/s)", file=sys.stderr)

    return embeddings, doc_map, model


def cosine_similarity(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb + 1e-10)


# ── RRF fusion ────────────────────────────────────────────────────


def rrf_fusion(bm25_scores, vec_scores, k=60, top_k=20):
    """Reciprocal rank fusion of BM25 and vector scores."""
    combined = defaultdict(float)

    # Get ranked lists
    bm25_ranked = sorted(bm25_scores.items(), key=lambda x: -x[1]) if bm25_scores else []
    vec_ranked = sorted(vec_scores.items(), key=lambda x: -x[1]) if vec_scores else []

    for i, (doc_id, _) in enumerate(bm25_ranked):
        combined[doc_id] += 1.0 / (k + i)

    for i, (doc_id, _) in enumerate(vec_ranked):
        combined[doc_id] += 1.0 / (k + i)

    return sorted(combined.items(), key=lambda x: -x[1])[:top_k]


# ── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", default=".repo-arch/index/hybrid")
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo)
    out_dir = os.path.join(repo_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load commits ──
    idx_path = os.path.join(repo_dir, ".repo-arch", "index", "full", "index.json")
    if not os.path.exists(idx_path):
        print("No full index. Run scripts/build_index.py first.", file=sys.stderr)
        sys.exit(1)

    with open(idx_path) as f: data = json.load(f)
    docs = data.get("documents", [])
    seen = set(); commits = []
    for d in docs:
        if d.get("sha","") not in seen:
            seen.add(d["sha"]); commits.append(d)
    print(f"Loaded {len(commits)} unique commits", file=sys.stderr)

    # ── Build embeddings ──
    embeddings, doc_map, model = build_embeddings(commits, args.model)

    # ── Build BM25 ──
    bm25 = BM25()
    for doc in commits:
        bm25.add(doc["id"], doc["text"])

    # ── Save index ──
    index_data = {
        "model": args.model,
        "commit_count": len(commits),
        "dimension": len(embeddings[0]) if len(embeddings) > 0 else 0,
    }

    # Save embeddings as numpy-compatible format
    import numpy as np
    emb_array = np.array(embeddings, dtype=np.float32)
    emb_path = os.path.join(out_dir, "embeddings.npy")
    np.save(emb_path, emb_array)
    print(f"Saved embeddings: {emb_path} ({emb_array.nbytes} bytes)", file=sys.stderr)

    # Save doc map (commit_id -> index)
    with open(os.path.join(out_dir, "doc_map.json"), "w") as f:
        json.dump(doc_map, f)

    # Save BM25 data
    bm25_data = {"documents": []}
    for doc in commits:
        bm25_data["documents"].append({
            "id": doc["id"],
            "sha": doc["sha"],
            "subject": doc.get("subject", ""),
            "packages": doc.get("packages", []),
            "paths": doc.get("paths", []),
            "date": doc.get("date", ""),
        })
    with open(os.path.join(out_dir, "docs.json"), "w") as f:
        json.dump(bm25_data, f)

    # Save metadata
    index_data["embeddings_file"] = "embeddings.npy"
    index_data["bm25_docs"] = len(commits)
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"\nIndex saved to {out_dir}", file=sys.stderr)
    print(f"  Embeddings: {len(embeddings)} x {len(embeddings[0])} ({emb_array.nbytes / 1e6:.1f} MB)", file=sys.stderr)
    print(f"  BM25 docs: {len(commits)}", file=sys.stderr)

    # ── Test search ──
    test_queries = [
        "What should I know about agent-session.ts?",
        "Which files co-change with models.generated.ts?",
        "What changed in the TUI package?",
    ]

    print(f"\nTest hybrid search:", file=sys.stderr)
    for query in test_queries:
        print(f"\n  Query: {query}", file=sys.stderr)

        # BM25
        bm25_scores = bm25.search_all_scores(query, k=200)
        bm25_top = sorted(bm25_scores.items(), key=lambda x: -x[1])[:5]

        # Vector
        q_emb = model.encode([query])[0]
        vec_scores = {}
        for i, emb in enumerate(embeddings):
            sim = cosine_similarity(q_emb, emb)
            vec_scores[commits[i]["id"]] = sim
        vec_top = sorted(vec_scores.items(), key=lambda x: -x[1])[:5]

        # Fused
        fused = rrf_fusion(bm25_scores, vec_scores, k=60, top_k=5)

        print(f"    BM25 top: {[(doc_id[:12], round(score, 3)) for doc_id, score in bm25_top]}", file=sys.stderr)
        print(f"    Vector top: {[(doc_id[:12], round(score, 3)) for doc_id, score in vec_top]}", file=sys.stderr)
        print(f"    Fused top: {[(doc_id[:12], round(score, 3)) for doc_id, score in fused]}", file=sys.stderr)

        # Show subjects
        def show_subjects(results, n=3):
            for doc_id, score in results[:n]:
                doc = next((d for d in commits if d["id"] == doc_id), None)
                if doc:
                    print(f"      {doc_id[:12]}: {doc.get('subject','')[:100]}", file=sys.stderr)

        print(f"    BM25:", file=sys.stderr); show_subjects(bm25_top)
        print(f"    Vector:", file=sys.stderr); show_subjects(vec_top)
        print(f"    Fused:", file=sys.stderr); show_subjects(fused)


if __name__ == "__main__":
    main()

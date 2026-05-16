#!/usr/bin/env python3
"""
Commit-cluster dataset generator: groups commits by file, generates Q&A per cluster.

Each file with multiple commits becomes a "mini-card" - a natural training example
for learning to aggregate commit history into answers.

Usage:
  source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
  python3 scripts/gen_cluster_dataset.py --target 2000 --out .repo-arch/training-data/clusters
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


# ── Load commit index ────────────────────────────────────────────


def load_commits(repo_dir):
    idx_path = os.path.join(repo_dir, ".repo-arch", "index", "full", "index.json")
    if not os.path.exists(idx_path):
        print("No index. Run scripts/build_index.py first.", file=sys.stderr)
        sys.exit(1)
    with open(idx_path) as f: data = json.load(f)
    docs = data.get("documents", [])
    seen = set(); unique = []
    for d in docs:
        if d.get("sha","") not in seen:
            seen.add(d["sha"]); unique.append(d)
    return unique


def load_cards(repo_dir):
    """Load accepted cards for additional context."""
    cfiles = glob.glob(os.path.join(repo_dir, ".repo-arch", "cache", "cards", "*.json"))
    if not cfiles: return []
    with open(cfiles[0]) as f: data = json.load(f)
    cards = data.get("cards", [])
    rev_path = os.path.join(repo_dir, ".repo-arch", "review-state.json")
    review = json.load(open(rev_path)) if os.path.exists(rev_path) else {}
    result = []
    for c in cards:
        cid = c.get("id", "")
        status = review.get(cid, {}).get("status", "unreviewed")
        if status == "accepted":
            result.append(c)
    return result


# ── Cluster commits by file ──────────────────────────────────────


def cluster_by_file(commits):
    """Group commits by file paths. Returns {file_path: [commits]}."""
    clusters = defaultdict(list)
    for doc in commits:
        for path in doc.get("paths", []):
            clusters[path].append(doc)
    # Sort each cluster by date
    for path in clusters:
        clusters[path].sort(key=lambda d: d.get("date", ""))
    return clusters


def get_package(path):
    m = re.match(r'(packages/[\w-]+)', path)
    return m.group(1) if m else "root"


# ── Generate questions per cluster ───────────────────────────────


def generate_cluster_questions(file_path, commits, package):
    """Generate diverse questions for a file's commit cluster."""
    n = len(commits)
    fix_count = sum(1 for c in commits if "fix" in c.get("subject","").lower())
    first_sha = commits[0].get("sha","")[:12]
    last_sha = commits[-1].get("sha","")[:12]
    first_date = commits[0].get("date","")
    last_date = commits[-1].get("date","")
    authors = list(set(c.get("author","") for c in commits if c.get("author")))
    subjects = [c.get("subject","") for c in commits]

    questions = []

    # Basic
    questions.append(f"How many commits touched {file_path}?")
    questions.append(f"What is the commit history of {file_path}?")
    questions.append(f"What changed in {file_path} between {first_sha} and {last_sha}?")

    # Package-aware
    if package and package != "root":
        questions.append(f"What changes were made to {package}/{os.path.basename(file_path)}?")
        questions.append(f"How has {file_path} evolved in the {package} package?")

    # Fix patterns
    if fix_count > 0:
        questions.append(f"How many fix commits affected {file_path}?")
        questions.append(f"What common bugs were fixed in {file_path}?")

    # Authors
    if len(authors) >= 2:
        questions.append(f"Who made changes to {file_path}?")
        questions.append(f"Which authors contributed most to {file_path}?")

    # Time range
    if first_date and last_date:
        questions.append(f"What is the oldest change in {file_path}?")

    # Subject patterns
    subjects_lower = " ".join(subjects).lower()
    if "refactor" in subjects_lower:
        questions.append(f"What refactors affected {file_path}?")
    if "test" in subjects_lower:
        questions.append(f"What test changes affected {file_path}?")
    if "revert" in subjects_lower:
        questions.append(f"What was reverted in {file_path}?")

    return questions, commits


def format_commit_cluster_context(file_path, commits, package, max_shown=8):
    """Format a file cluster into a readable context block."""
    lines = [
        f"--- FILE: {file_path} ---",
        f"  Package: {package}",
        f"  Total commits: {len(commits)}",
    ]

    for c in commits[:max_shown]:
        sha = c.get("sha","")[:12]
        subject = c.get("subject","")[:150]
        date = c.get("date","").split("T")[0] if c.get("date") else ""
        author = c.get("author","")
        files_str = ""
        paths = c.get("paths", [])
        if len(paths) > 1:
            files_str = f" ({len(paths)} files)"
        lines.append(f"  {date} {sha} {author}: {subject}{files_str}")

    if len(commits) > max_shown:
        lines.append(f"  ... and {len(commits) - max_shown} more commits")

    return "\n".join(lines)


# ── Generate answer ──────────────────────────────────────────────


def generate_answer(context_str, question, max_tokens=384, temperature=0.2):
    """Use base model to generate reference answer."""
    user_prompt = f"### Context\n{context_str}\n\n### Question\n{question}\n\n### Answer"
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
        "--system-prompt", "Answer concisely using the context above. Cite specific file paths, numbers, and commit SHAs.",
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
        return answer.strip() or "[empty]"
    except Exception as e:
        return f"[error: {e}]"


def format_example(context_str, question, answer):
    """MLX-compatible chat format."""
    user_content = (
        "Analyze the following file commit history. "
        "Answer based only on this evidence.\n\n"
        f"{context_str}\n\n"
        f"### Question\n{question}\n\n"
        "### Answer"
    )
    return {"messages": [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]}


# ── Negatives ────────────────────────────────────────────────────


def generate_cluster_negatives(count=100):
    """Questions that no single file cluster can answer."""
    base = [
        "What is the project's overall architecture?",
        "How does the CI/CD pipeline work?",
        "What is the project's release cadence?",
        "How are security vulnerabilities handled?",
        "What is the database schema?",
        "How is performance benchmarked?",
        "What third-party services does the project depend on?",
        "How are feature flags managed in production?",
        "What is the API rate limiting strategy?",
        "How is documentation published?",
        "What is the testing pyramid for this project?",
        "How are dependency updates reviewed?",
        "What is the disaster recovery plan?",
        "How are on-call rotations managed?",
        "What is the project's SLA?",
        "How is data encrypted at rest?",
        "What monitoring dashboards exist?",
        "How are A/B tests designed?",
        "What is the cost structure for cloud resources?",
        "How are user permissions managed?",
    ]
    return base[:count]


# ── Main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--out", default=".repo-arch/training-data/clusters")
    parser.add_argument("--min-commits-per-file", type=int, default=3, help="Min commits to make a cluster")
    parser.add_argument("--max-clusters", type=int, default=200, help="Max file clusters to use")
    parser.add_argument("--questions-per-cluster", type=int, default=3)
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true", help="Count available examples without generating")
    args = parser.parse_args()

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_dir, args.out)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading commits...", file=sys.stderr)
    commits = load_commits(repo_dir)
    print(f"  {len(commits)} unique commits", file=sys.stderr)

    # Cluster by file
    clusters = cluster_by_file(commits)
    # Filter: only files with >= min commits
    viable = {f: cs for f, cs in clusters.items() if len(cs) >= args.min_commits_per_file}
    # Skip CHANGELOGs and index files (too noisy)
    viable = {f: cs for f, cs in viable.items() if "CHANGELOG" not in f and "/index." not in f}

    print(f"  {len(clusters)} total files, {len(viable)} with >= {args.min_commits_per_file} commits", file=sys.stderr)

    # Show distribution
    commit_counts = sorted([len(cs) for cs in viable.values()], reverse=True)
    print(f"  Top file: {commit_counts[0] if commit_counts else 0} commits", file=sys.stderr)
    print(f"  Median:   {commit_counts[len(commit_counts)//2] if commit_counts else 0} commits", file=sys.stderr)

    # Potential max examples
    max_clusters = min(args.max_clusters, len(viable))
    potential = max_clusters * args.questions_per_cluster
    print(f"  Max clusters: {max_clusters}, questions per cluster: {args.questions_per_cluster}", file=sys.stderr)
    print(f"  Potential examples: {potential} (plus {int(potential * args.negative_ratio)} negatives)", file=sys.stderr)

    # Pick top clusters by commit count (highest signal)
    sorted_files = sorted(viable.keys(), key=lambda f: -len(viable[f]))
    selected_files = sorted_files[:max_clusters]

    if args.dry_run:
        # Show some examples of what would be generated
        print(f"\nSample clusters:", file=sys.stderr)
        for f in selected_files[:5]:
            cs = viable[f]
            pkg = get_package(f)
            authors = list(set(c.get("author","") for c in cs if c.get("author")))
            fix_c = sum(1 for c in cs if "fix" in c.get("subject","").lower())
            print(f"  {f} [{pkg}]", file=sys.stderr)
            print(f"    {len(cs)} commits, {len(authors)} authors, {fix_c} fixes", file=sys.stderr)
            print(f"    Range: {cs[0].get('date','')[:10]} -> {cs[-1].get('date','')[:10]}", file=sys.stderr)
        print(f"\nDry run complete. Would generate ~{potential} examples.", file=sys.stderr)
        return

    # ── Generate examples ──
    examples = []
    total_potential = max_clusters * args.questions_per_cluster
    neg_count = int(total_potential * args.negative_ratio / (1 + args.negative_ratio))
    pos_count = total_potential - neg_count

    genned = 0
    for file_path in selected_files:
        cluster_commits = viable[file_path]
        package = get_package(file_path)
        questions, _ = generate_cluster_questions(file_path, cluster_commits, package)

        for q in questions[:args.questions_per_cluster]:
            context_str = format_commit_cluster_context(file_path, cluster_commits, package)
            answer = generate_answer(context_str, q)
            example = format_example(context_str, q, answer)
            example["source"] = f"cluster:{file_path[:60]}"
            examples.append(example)
            genned += 1

            if genned % 30 == 0:
                print(f"  Generated {genned}/{pos_count}...", file=sys.stderr)
            if genned >= pos_count:
                break
        if genned >= pos_count:
            break

    # Negatives
    neg_questions = generate_cluster_negatives(count=neg_count)
    for q in neg_questions:
        answer = (
            "The available file-level commit history does not cover this topic. "
            "The context only covers individual file changes, not cross-cutting concerns "
            "like architecture, CI/CD, or deployment."
        )
        example = format_example("No relevant file history found for this question.", q, answer)
        example["source"] = "negative"
        examples.append(example)

    # Shuffle and split
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
        "total_examples": len(examples),
        "train": len(train),
        "valid": len(valid),
        "clusters_used": min(genned // args.questions_per_cluster, len(selected_files)),
        "max_clusters": max_clusters,
        "questions_per_cluster": args.questions_per_cluster,
        "min_commits_per_file": args.min_commits_per_file,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata: {json.dumps(meta, indent=2)}", file=sys.stderr)

    if examples:
        print(f"\n=== Sample ===\n{json.dumps(examples[0], indent=2)[:1200]}", file=sys.stderr)


if __name__ == "__main__":
    main()

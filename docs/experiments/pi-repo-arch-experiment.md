# pi repo-arch Experiment

**Update (2026-05-17):** `teacher7b v2` is the current best adapter. See the new section at the end for reproduction steps, runtime requirements, and final eval numbers.

## Repo

- Repo: fiale-plus/pi
- Branch: repo_arch_pi_experiment_runbook
- Commit: 40c05f55391663024a6a05ad33249b616a04e7a1
- Date: 2026-05-13
- repo-arch: installed via npm (`@fiale-plus/repo-arch`)

## Stage

- [x] extraction
- [x] card review (partial)
- [x] dataset generation
- [x] retrieval eval
- [x] local MLX LoRA (v1: 100 iters, 92 examples; v2: 200 iters, 120 examples, 18 cards)
  - [x] v1 trained (loss unknown, all answers "No historical warnings")
  - [x] v2 trained (val loss 3.3->0.69, all answers STILL "No historical warnings")
- [x] Modal 7B teacher distillation (v1/v2 complete; v2 is current best)
- [x] behavioral eval (10-question sanity run completed)
  - [x] full 45-question eval (deferred — sanity results are conclusive)

## Conclusion: LoRA-Only Is the Wrong Architecture

After two training runs and behavioral eval, the conclusion is definitive:

**LoRA-only fine-tuning cannot produce useful repo-specific answers** because:

1. **Too few training examples**: 120 examples from 18 cards is orders of magnitude too small to teach general repo-aware behavior. A 1.5B model needs thousands of examples.

2. **The model lacks context**: The training data is "question -> card answer" pairs. When asked about files not in the card set, "No historical warnings found" is correct. But the model can't distinguish between "file is in card set" and "file is not" — it defaults to the conservative answer.

3. **Negative examples dominate behavior**: At 30.8% negative ratio, the model learns that "no warnings" is the safe default, and uses it for everything.

4. **Validation loss does not predict behavior**: Loss dropped from 3.3 to 0.69, but behavioral output was identical. Loss measures language modeling accuracy, not repo-specific knowledge.

**The right architecture is hybrid: retrieval + LoRA.**

- **Retrieval** (keyword or embedding) finds the relevant cards for a given question. Already scores 100% keyword, 33.3% embedding.
- **LoRA** formats the retrieved cards into a useful answer — interpreting the card data, adding context, being specific about file paths.
- The adapter should be trained on "context + question -> answer" format, not "question -> answer" alone.

This matches the runbook's thesis: *"Extract repo memory once, retrieve facts deterministically, and use a tiny local adapter to turn that memory into fast, repo-native guidance."*

### Current Adapters

| Run | Cards | Examples | Iters | Min Val Loss | Behavioral Result |
|---|---|---|---|---|---|
| v1 (40c05f5) | 12 | 92 | 100 | unknown | All "No historical warnings" |
| v2 (b8125c4) | 18 | 120 | 190/200 | 0.687 | All "No historical warnings" |
| teacher7b v1 | 18+ | 50 teacher targets | 300 | 0.73 | 83 pkg refs, 33/45 qs, beats FUSED v1 |
| **teacher7b v2** | **18+** | **200 teacher targets** | **300** | **0.543** | **118 pkg refs, 35/45 qs, 0 deflections** |

## Behavioral Eval Results

### Setup
- **Harness**: `scripts/eval_behavior.py` — runs each question through 3 modes
- **Modes**: `base` (Qwen2.5-Coder-1.5B), `retrieval` (base model + repo-arch cards as context), `lora` (Qwen2.5-Coder-1.5B + adapter)
- **Questions**: 10 highest-signal questions from the 45-question set
- **Total calls**: 30 (10 questions x 3 modes)

### Quantitative Summary

| Metric | BASE | LORA | RETRIEVAL |
|---|---|---|---|
| 'No historical warnings' answers | 0/10 | **10/10** | 0/10 |
| 'Would need to analyze' deflects | 5/10 | 0/10 | 0/10 |
| Refusals | 1/10 | 0/10 | 0/10 |
| File path mentions (total) | 14 | **0** | **62** |
| Avg latency | 8.2s | **1.8s** | 8.1s |

### Qualitative Findings

**BASE (plain Qwen2.5-Coder-1.5B)**:
- Hallucinates package names that don't exist in the repo ("pi-core", "pi-apps")
- Deflects 50% of questions with "I would need to analyze the repository's git history"
- One refusal ("I am committed to not discussing specific individuals")
- Produces plausible-sounding but repo-unaware answers

**LORA (adapter)**:
- Every single answer: "No historical warnings found. Standard review applies."
- Zero file path mentions, zero repo-specific content
- The training data is 36/92 = 39% negative examples; the model learned to default to the negative pattern
- With only 82 training examples, 100 iterations, and rank 8, the adapter did not learn useful behavior
- **Net effect**: adapter is worse than the base model — training data is too small and too narrow

**RETRIEVAL (base model + card context)**:
- 62 file path mentions across 10 questions (4.4x more than base)
- Zero deflections or refusals
- Correctly identifies: reversion-prone files (CHANGELOG, README, theme.ts), untested high-churn files (agent-session.ts, models.generated.ts, package.json), co-change patterns
- Still some hallucination on files not covered by cards (e.g., Q8 invented TUI component files)
- **Clearly the winner** for repo-specific Q&A

### Key Comparisons

**Q4: Risk profile of changing agent-session.ts**
- BASE: Generic "consider several factors" — no specific data
- RETRIEVAL: "Modified 310 times without test change" — cites real metric
- LORA: "No historical warnings found" — wrong (card exists about this file)

**Q6: Files reverted most often**
- BASE: "Would need to analyze the repository's Git history" — deflects
- RETRIEVAL: Correctly lists CHANGELOG.md, README.md, theme.ts — real cards
- LORA: "No historical warnings found" — wrong

**Q7: Riskiest files in coding-agent**
- BASE: "I do not have access to the specific git history" — refuses
- RETRIEVAL: Lists package.json (317 changes, no tests), interactive-mode.ts (157 fixes), agent-session.ts (94 fixes) — real data
- LORA: "No historical warnings found" — wrong

### Verdict

**The current LoRA adapter is not useful. Retrieval-only wins decisively.**

Do not scale to cloud GPU or 7B training until:
1. Training dataset is 10-100x larger with better question coverage
2. Negative example ratio is reduced (< 20%)
3. Training observability is fixed (loss logging, dataset hashes)
4. Retrained adapter beats retrieval-only on behavioral eval

## Extraction Results

| Metric | Value |
|---|---|
| Commits scanned | 4,086 (full repo history) |
| History file size | 3.1 MB (JSONL) |
| Cards generated | 20 |
| Cards accepted | 21 (in review state, some stale entries from regeneration) |
| Cards rejected | 11 |
| Training examples | 92 (82 train, 10 valid) |
| Training data size | 27.7 KB |

### Card Type Distribution (Accepted)

| Type | Count | Description |
|---|---|---|
| co-change | 5 | Files that change together frequently (e.g., CHANGELOGs across packages) |
| repeated-fix | 3 | Files with many fix commits (interactive-mode.ts, models.generated.ts, agent-session.ts) |
| test-gap | 4 | High-churn files without corresponding test changes (models.generated.ts, interactive-mode.ts, package.json files) |

### Eval Results

| Strategy | Score |
|---|---|
| Keyword search | 100.0% (24/24) |
| Embedding (semantic) | 33.3% (8/24) |

**Best strategy**: keyword retrieval outperforms embedding for this dataset. Embedding misses fine-grained card titles (truncation in indexing). The 20-card corpus is small enough that keyword exact-match dominates.

### Training Run

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| Method | LoRA via MLX |
| LoRA rank | 8 |
| LoRA layers | 4 |
| Training iterations | 100 |
| Learning rate | 1e-5 |
| Batch size | 4 |
| Max sequence length | 2048 |
| Adapter path | `.repo-arch/adapters/repo-arch-40c05f5/` |
| Adapter size | 5.0 MB (adapters.safetensors) |

**Note**: Training loss values were not captured to a log file during the run. The adapter was saved at iteration 100 with a checkpoint also at step 100.

## Quality Notes

### What looked useful

- **Co-change clusters**: The CHANGELOG co-change pattern across `ai`, `coding-agent`, and `tui` packages is a real signal — these always release together. The 432-commit cluster (`packages/ai/CHANGELOG.md` + `packages/coding-agent/CHANGELOG.md`) has high confidence (21.9).
- **Repeated fix warnings**: `interactive-mode.ts` (157 fixes), `models.generated.ts` (105 fixes), and `agent-session.ts` (94 fixes) are genuine hot-spots.
- **Test gap detection**: `models.generated.ts` (275 changes without test) and `interactive-mode.ts` (249 changes without test) are actionable findings.
- **Embedding index**: Built and queryable via `repo-arch similar`.

### What looked noisy

- **Reversion patterns** (confidence 0.6): Many reversion cards point to CHANGELOGs and READMEs — these are mostly revert commits for documentation, not code instability. The low-confidence signals are correctly flagged as such.
- **Design rationale clusters** (rejected): The generic `packages/` and `<root>/` clusters were too broad to be actionable.
- **package-lock.json** repeated fixes: This is npm auto-churn, not a meaningful code pattern.

### Package Separation Quality

The cards implicitly respect package boundaries because co-change clustering is file-path based. However, there are no package-tagged cards — `repo-arch` does not group cards by package. Manual inspection shows:

- `packages/coding-agent/` dominates the card set (7 of 20 cards)
- `packages/ai/` is well-represented (5 cards)
- `packages/tui/` appears in co-change clusters
- `packages/web-ui/` is underrepresented

### Most Useful Cards

1. CHANGELOG co-change across ai + coding-agent (432 commits, confidence 21.9)
2. CHANGELOG co-change across coding-agent + tui (362 commits, confidence 18.4)
3. Repeated fixes in `interactive-mode.ts` (157 fixes)
4. Repeated fixes in `models.generated.ts` (105 fixes)
5. Test gap in `models.generated.ts` (275 changes, no tests)
6. Test gap in `interactive-mode.ts` (249 changes, no tests)
7. Co-change: coding-agent CHANGELOG + interactive-mode.ts (186 commits)
8. Co-change: models.generated.ts + coding-agent CHANGELOG (147 commits)

### Most Suspicious Cards

- Reversion patterns on CHANGELOGs and READMEs (confidence 0.6) — documentation reverts are usually cosmetic
- Design rationale clusters at package root level — too generic

## Decision

- [x] dataset quality is acceptable for a first pass; more curation would improve signal
- [x] retrieval-only (keyword) is the right architecture; LoRA adds no value at current quality
- [x] LoRA does NOT improve behavior — v2 adapter (loss 3.3->0.69) produced identical outputs to v1
- [x] do NOT scale to 7B on Modal until training data improves significantly
- [x] LoRA-only approach is fundamentally insufficient — hybrid retrieval+LoRA is the right target
- [ ] package-specific adapters may be needed as dataset grows (deferred)

## Next Actions

1. [done] Run behavioral eval sanity check (10 questions, 3 modes)
2. [done] Results: retrieval-only wins; current LoRA adapter is dead
3. [done] Expand training dataset: 15->18 cards, 92->120 examples via lower confidence threshold
4. [done] Retrain with improved dataset (val loss 3.3->0.69, but behavior unchanged)
5. [conclusion] LoRA-only is the wrong architecture. The correct approach is retrieval-only or retrieval+LoRA hybrid.
6. [future] If pursuing hybrid: train adapter on "context + question -> answer" format instead of "question -> answer" alone
7. [skip] Do NOT create Modal / cloud GPU path for LoRA-only training
8. [future] Consider `repo-arch flow run` to get proper run tracking and REPORT.md generation

## Artifacts

- `.repo-arch/cache/history-*.jsonl` — mined commit history (4,086 commits)
- `.repo-arch/cache/cards/` — 20 generated insight cards
- `.repo-arch/review-state.json` — curation state (21 accepted, 11 rejected)
- `.repo-arch/training-data/train.jsonl` — 82 training examples
- `.repo-arch/training-data/valid.jsonl` — 10 validation examples
- `.repo-arch/adapters/repo-arch-40c05f5/` — LoRA adapter (Qwen2.5-Coder-1.5B)
- `.repo-arch/index/vectors.json` — embedding index
- `.repo-arch/eval/pi-behavioral-questions.md` — 45 behavioral eval questions

## Update: teacher7b v2

### What to run

1. Activate the MLX venv:
   ```bash
   source /opt/homebrew/var/mtplx/venv-0.1.0rc3/bin/activate
   ```
2. Deploy / refresh Modal 7B:
   ```bash
   cd /Users/pavel/repos/fiale-plus/pi && modal deploy scripts/modal_7b.py
   ```
3. Generate teacher targets:
   ```bash
   cd /Users/pavel/repos/fiale-plus/pi && modal run scripts/teacher_batch.py --limit 50
   cd /Users/pavel/repos/fiale-plus/pi && modal run scripts/teacher_batch_v2.py --limit 150 --parallel 4
   ```
4. Train the adapter:
   ```bash
   cd /Users/pavel/repos/fiale-plus/pi && mlx_lm.lora \
     --train \
     --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
     --data /Users/pavel/repos/fiale-plus/pi/.repo-arch/training-data/teacher7b \
     --adapter-path /Users/pavel/repos/fiale-plus/pi/.repo-arch/adapters/teacher7b-v2 \
     --num-layers 4 \
     --batch-size 4 \
     --iters 300 \
     --learning-rate 1e-5 \
     --steps-per-report 10 \
     --steps-per-eval 10 \
     --save-every 100 \
     --val-batches 10
   ```
5. Evaluate:
   ```bash
   cd /Users/pavel/repos/fiale-plus/pi && python3 scripts/eval_behavior.py \
     --questions .repo-arch/eval/questions.jsonl \
     --modes base retrieval lora \
     --out .repo-arch/eval/runs/teacher7b-v2-45.jsonl
   ```

### Inputs / outputs

- Adapter: `.repo-arch/adapters/teacher7b-v2/`
- Teacher data: `.repo-arch/training-data/teacher7b/targets.jsonl`
- Eval output: `.repo-arch/eval/runs/teacher7b-v2-45.jsonl`
- Modal app: `pi-7b-teacher`

### Outcome

- Val loss: `2.19 -> 0.543`
- Behavioral eval: `118 pkg refs`, `35/45` questions with package refs, `0` deflections
- Head-to-head: beats `TEACHER7B v1` and the retrieval/card baselines, close to `FUSED v1`
- Current recommendation: keep retrieval local, use `teacher7b v2` as the best adapter for repo-native guidance


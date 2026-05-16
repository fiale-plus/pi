# CI Pipeline: Recurring repo Mining → Teacher 7B → Distilled Adapter

## Architecture

```
Weekly cron / commit trigger
  → repo-arch mine + classify + cards
  → content-hash check (skip if unchanged)
  → Modal 7B teacher generation (A10G, ~$0.15/batch)
  → local MLX LoRA distillation (1-2B model)
  → 45+ question behavioral eval gate
  → publish blessed adapter
```

## Modal Usage

**Use `modal.Function` (raw Python scripts), not notebooks or web endpoints.**

| Approach | Fit | Our choice |
|---|---|---|
| `modal.Function` | Batch inference, CI-friendly, scriptable | ✅ |
| Notebooks | Exploration only | ❌ |
| Web endpoints | Interactive serving | ❌ |

**Key patterns:**
- `modal.Function.from_name("pi-7b-teacher", "generate")` to reference deployed function
- `modal deploy scripts/modal_7b.py` before calling from other scripts
- Pin `modal==1.4.2` in Image to avoid API churn
- Use `Function.map()` for parallel batch generation

## Caching

Cache key = `hash(repo_sha + miner_ver + prompt_ver + teacher_model)`

Skip teacher generation if cache hit. Store past adapters for rollback.

## Eval Gate

Compare new adapter vs current blessed on 45+ questions:
- Package refs, SHA refs, deflections
- Head-to-head win/loss
- Zero regressions

## Cost

- Modal A10G: ~$0.60/hr
- 50 questions sequential: ~$0.15
- 50 questions parallel (Function.map): ~$0.02
- Local distillation: free (Mac MLX)

## Files

- `scripts/modal_7b.py` — Modal 7B deployment
- `scripts/teacher_batch.py` — Batch teacher generation
- `scripts/teacher_gen.py` — Generate batch questions + context
- `.repo-arch/adapters/teacher7b/` — Latest teacher-distilled adapter
- `.repo-arch/training-data/teacher7b/` — Training data

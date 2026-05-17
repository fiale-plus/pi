# pi Behavioral Eval Questions

Fixed question set for comparing base model, retrieval-only, LoRA-only, and retrieval+LoRA.

## Package Boundaries

1. Which packages tend to co-change together, and what does that imply about their coupling?
2. When modifying `packages/ai/src/models.generated.ts`, which other files should you inspect?
3. What architectural decisions are visible around the unified LLM provider layer?
4. What is the risk profile of changing `packages/coding-agent/src/core/agent-session.ts`?
5. Which package boundaries are the most fragile based on historical breakages?

## Risky Files

6. Which files have been reverted most often, and why?
7. What are the riskiest files to modify in the coding agent package?
8. Which files in the TUI package have required repeated fixes?
9. What files in the AI package have no test coverage despite high change frequency?
10. Which configuration files change without corresponding test updates?

## Historical Patterns

11. What historical breakages should I know before changing the provider routing?
12. Where has CLI state/config handling been fragile?
13. What changed repeatedly in the TUI package and why?
14. What web UI changes historically required backend or agent changes?
15. What is the reversion history of the coding agent README?
16. How has the changelog management pattern evolved across packages?

## Test Coverage

17. Which packages have the most untested high-churn files?
18. What would you tell a new contributor about test expectations before touching package boundaries?
19. Which files have been fixed most often without corresponding test additions?
20. Are there test gaps in the AI provider layer specifically?

## Coding Agent Internals

21. What should a coding agent inspect before modifying provider-tool-calling behavior?
22. What is the relationship between interactive mode and theme configuration historically?
23. How fragile is the agent session lifecycle based on commit patterns?
24. What modes have been added or removed from the coding agent?
25. What is the history of model resolver changes?

## AI Provider Layer

26. How many LLM providers are supported and how are they registered?
27. What is the architectural pattern for adding a new provider?
28. Which provider has the most complex auth requirements historically?
29. How have streaming options evolved across providers?
30. What is the relationship between model generation scripts and the type system?

## TUI/UI

31. What are the most frequently changed TUI components?
32. How has the keybinding system evolved?
33. What theme-related changes have been made and reverted?
34. What is the history of the prompt input handling?
35. How has error display evolved in the TUI?

## Root/Infrastructure

36. What npm scripts and config are essential for the build pipeline?
37. How has the CI configuration evolved?
38. What is the history of dependency management (package-lock churn)?
39. Which root-level config files are most frequently modified and why?
40. What release process conventions are visible in the commit history?

## Cross-Cutting

41. What is the relationship between CHANGELOG updates and actual code changes across packages?
42. How do model updates cascade through the system?
43. What is the most common cause of bugs based on fix commit patterns?
44. How does the system handle provider API changes over time?
45. What architectural debt is visible from the commit history?
